# ======================== Azure Infrastructure Setup Script ========================
# PowerShell script to set up all Azure resources for ADAS Model API
# 
# Prerequisites:
#   1. Azure CLI installed: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows
#   2. Logged in to Azure: az login
#   3. Python and pip installed
#
# Usage:
#   .\azure_setup.ps1
#
# The script will prompt for necessary information.

# ======================== Configuration ========================

param(
    [string]$SubscriptionId = "",
    [string]$ResourceGroupName = "adas-ml-rg",
    [string]$Location = "eastus",
    [string]$WorkspaceName = "adas-ml-ws",
    [string]$StorageAccountName = "",
    [string]$ContainerInstanceName = "adas-api-container",
    [string]$ContainerRegistryName = "",
    [switch]$SkipAzureML = $false,
    [switch]$SkipContainerRegistry = $false
)

# ======================== Helper Functions ========================

function Write-Header {
    param([string]$Message)
    Write-Host "`n" + "="*70 -ForegroundColor Cyan
    Write-Host "▶ $Message" -ForegroundColor Cyan
    Write-Host "="*70 -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Message)
    Write-Host "  ┌─ $Message" -ForegroundColor Green
}

function Write-Success {
    param([string]$Message)
    Write-Host "  ✓ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "  ⚠ $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "  ✗ $Message" -ForegroundColor Red
}

function Test-AzureCLI {
    try {
        $version = az version --output json | ConvertFrom-Json
        Write-Success "Azure CLI is installed (version: $($version.'azure-cli'))"
        return $true
    }
    catch {
        Write-Error "Azure CLI is not installed or not in PATH"
        Write-Host "  Install from: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows"
        return $false
    }
}

function Test-AzureLogin {
    try {
        $account = az account show --output json | ConvertFrom-Json
        Write-Success "Logged in as: $($account.user.name)"
        Write-Success "Subscription: $($account.name) ($($account.id))"
        return $true
    }
    catch {
        return $false
    }
}

function Get-RandomString {
    param([int]$Length = 6)
    $chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    $random = -join ((0..$Length) | ForEach-Object { $chars[(Get-Random -Maximum $chars.Length)] })
    return $random
}

# ======================== Pre-flight Checks ========================

Write-Header "Pre-flight Checks"

Write-Step "Checking Azure CLI installation"
if (-not (Test-AzureCLI)) {
    exit 1
}

Write-Step "Checking Azure login status"
if (-not (Test-AzureLogin)) {
    Write-Host "  Please run: az login"
    exit 1
}

# Get current subscription
$currentSub = az account show --output json | ConvertFrom-Json
if ([string]::IsNullOrEmpty($SubscriptionId)) {
    $SubscriptionId = $currentSub.id
}

Write-Success "Ready to provision resources"

# ======================== Generate Names ========================

Write-Header "Generating Resource Names"

$randomSuffix = Get-RandomString 6

if ([string]::IsNullOrEmpty($StorageAccountName)) {
    # Storage account names must be globally unique and lowercase
    $StorageAccountName = "adsastorage$randomSuffix"
}

if ([string]::IsNullOrEmpty($ContainerRegistryName)) {
    # Container registry names must be globally unique and lowercase
    $ContainerRegistryName = "adasregistry$randomSuffix"
}

Write-Success "Resource Group: $ResourceGroupName"
Write-Success "ML Workspace: $WorkspaceName"
Write-Success "Storage Account: $StorageAccountName"
Write-Success "Container Registry: $ContainerRegistryName"
Write-Success "Container Instance: $ContainerInstanceName"
Write-Success "Location: $Location"

# ======================== Create Resource Group ========================

Write-Header "Creating Resource Group"

Write-Step "Creating resource group: $ResourceGroupName"
try {
    az group create `
        --name $ResourceGroupName `
        --location $Location `
        --output none
    Write-Success "Resource group created"
}
catch {
    Write-Error "Failed to create resource group"
    exit 1
}

# ======================== Create Storage Account ========================

Write-Header "Creating Storage Account"

Write-Step "Creating storage account: $StorageAccountName"
try {
    az storage account create `
        --name $StorageAccountName `
        --resource-group $ResourceGroupName `
        --location $Location `
        --sku Standard_LRS `
        --kind StorageV2 `
        --output none
    Write-Success "Storage account created"
}
catch {
    Write-Error "Failed to create storage account"
    exit 1
}

Write-Step "Creating blob container: models"
try {
    $storageKey = az storage account keys list `
        --account-name $StorageAccountName `
        --resource-group $ResourceGroupName `
        --output json | ConvertFrom-Json | Select-Object -ExpandProperty key -First 1
    
    az storage container create `
        --account-name $StorageAccountName `
        --name models `
        --account-key $storageKey.value `
        --output none
    Write-Success "Blob container created"
}
catch {
    Write-Error "Failed to create blob container"
}

# ======================== Create Azure ML Workspace ========================

if (-not $SkipAzureML) {
    Write-Header "Creating Azure ML Workspace"
    
    Write-Step "Creating ML workspace: $WorkspaceName"
    try {
        az ml workspace create `
            --name $WorkspaceName `
            --resource-group $ResourceGroupName `
            --location $Location `
            --storage-account $StorageAccountName `
            --output none
        Write-Success "ML workspace created"
    }
    catch {
        Write-Warning "Azure ML workspace creation skipped (might require additional setup)"
    }
    
    Write-Step "Getting workspace details"
    try {
        $workspace = az ml workspace show `
            --name $WorkspaceName `
            --resource-group $ResourceGroupName `
            --output json | ConvertFrom-Json
        Write-Success "Workspace ID: $($workspace.id)"
    }
    catch {
        Write-Warning "Could not retrieve workspace details"
    }
}
else {
    Write-Warning "Skipped Azure ML Workspace creation"
}

# ======================== Create Container Registry ========================

if (-not $SkipContainerRegistry) {
    Write-Header "Creating Container Registry"
    
    Write-Step "Creating container registry: $ContainerRegistryName"
    try {
        az acr create `
            --name $ContainerRegistryName `
            --resource-group $ResourceGroupName `
            --sku Basic `
            --admin-enabled true `
            --output none
        Write-Success "Container registry created"
    }
    catch {
        Write-Error "Failed to create container registry"
    }
    
    Write-Step "Getting registry credentials"
    try {
        $registryUrl = az acr show `
            --name $ContainerRegistryName `
            --resource-group $ResourceGroupName `
            --query loginServer --output tsv
        
        $registryCreds = az acr credential show `
            --name $ContainerRegistryName `
            --resource-group $ResourceGroupName `
            --output json | ConvertFrom-Json
        
        Write-Success "Registry URL: $registryUrl"
        Write-Success "Username: $($registryCreds.username)"
    }
    catch {
        Write-Warning "Could not retrieve container registry credentials"
    }
}
else {
    Write-Warning "Skipped Container Registry creation"
}

# ======================== Create Key Vault (Optional) ========================

Write-Header "Creating Key Vault (Optional)"

$keyVaultName = "adas-keyvault-$randomSuffix"

Write-Step "Creating Key Vault: $keyVaultName"
try {
    az keyvault create `
        --name $keyVaultName `
        --resource-group $ResourceGroupName `
        --location $Location `
        --output none
    Write-Success "Key Vault created"
    
    # Store connection strings
    Write-Step "Storing secrets in Key Vault"
    
    $storageConnStr = az storage account show-connection-string `
        --name $StorageAccountName `
        --resource-group $ResourceGroupName `
        --output tsv
    
    az keyvault secret set `
        --vault-name $keyVaultName `
        --name "storage-connection-string" `
        --value $storageConnStr `
        --output none
    Write-Success "Stored storage connection string"
}
catch {
    Write-Warning "Could not create Key Vault"
}

# ======================== Prepare Dockerfile in Azure ========================

Write-Header "Preparing for Container Deployment"

Write-Step "Building Docker image for Azure Container Registry"
Write-Host "  Command to run:"
Write-Host "  az acr build --registry $ContainerRegistryName --image adas-api:latest ."

# ======================== Summary ========================

Write-Header "Setup Summary"

Write-Success "✅ Resource Group: $ResourceGroupName"
Write-Success "✅ Storage Account: $StorageAccountName"
Write-Success "✅ Container Registry: $ContainerRegistryName"
Write-Success "✅ ML Workspace: $WorkspaceName"
Write-Success "✅ Key Vault: $keyVaultName"

Write-Host "`n📝 Next Steps:"
Write-Host "`n1. BUILD AND PUSH DOCKER IMAGE:"
Write-Host "   cd <project-directory>"
Write-Host "   az acr build --registry $ContainerRegistryName --image adas-api:latest ."
Write-Host "`n2. DEPLOY TO AZURE CONTAINER INSTANCES:"
Write-Host "   az container create \"
Write-Host "     --resource-group $ResourceGroupName \"
Write-Host "     --name $ContainerInstanceName \"
Write-Host "     --image $ContainerRegistryName.azurecr.io/adas-api:latest \"
Write-Host "     --registry-username (az acr credential show --name $ContainerRegistryName --query username) \"
Write-Host "     --registry-password (az acr credential show --name $ContainerRegistryName --query passwords[0].value) \"
Write-Host "     --ports 8000 8501 \"
Write-Host "     --cpu 2 --memory 4"
Write-Host "`n3. VIEW DEPLOYMENT:"
Write-Host "   az container show --resource-group $ResourceGroupName --name $ContainerInstanceName"
Write-Host "`n4. VIEW LOGS:"
Write-Host "   az container logs --resource-group $ResourceGroupName --name $ContainerInstanceName"
Write-Host "`n5. DELETE RESOURCES (when done):"
Write-Host "   az group delete --name $ResourceGroupName --yes --no-wait"

Write-Host "`n📋 SAVE THESE VALUES FOR LATER:"
Write-Host "   Subscription: $SubscriptionId"
Write-Host "   Resource Group: $ResourceGroupName"
Write-Host "   Storage Account: $StorageAccountName"
Write-Host "   Container Registry: $ContainerRegistryName"
Write-Host "   ML Workspace: $WorkspaceName"

Write-Host "`n✅ Setup complete!`n"
