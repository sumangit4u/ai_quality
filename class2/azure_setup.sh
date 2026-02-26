#!/bin/bash

# ======================== Azure Infrastructure Setup Script ========================
# Bash script to set up all Azure resources for ADAS Model API
# 
# Prerequisites:
#   1. Azure CLI installed: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
#   2. Logged in to Azure: az login
#   3. jq installed for JSON parsing: sudo apt-get install jq (or brew install jq)
#
# Usage:
#   chmod +x azure_setup.sh
#   ./azure_setup.sh
#
# The script will prompt for necessary information.

set -e

# ======================== Configuration ========================

SUBSCRIPTION_ID="${1:-}"
RESOURCE_GROUP_NAME="${2:-adas-ml-rg}"
LOCATION="${3:-eastus}"
WORKSPACE_NAME="${4:-adas-ml-ws}"
STORAGE_ACCOUNT_NAME="${5:-}"
CONTAINER_INSTANCE_NAME="adas-api-container"

RANDOM_SUFFIX=$(openssl rand -hex 3)

if [ -z "$STORAGE_ACCOUNT_NAME" ]; then
    STORAGE_ACCOUNT_NAME="adsastorage${RANDOM_SUFFIX}"
fi

CONTAINER_REGISTRY_NAME="adasregistry${RANDOM_SUFFIX}"

# ======================== Color Codes ========================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ======================== Helper Functions ========================

print_header() {
    echo -e "\n${CYAN}$(printf '=%.0s' {1..70})${NC}"
    echo -e "${CYAN}▶ $1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' {1..70})${NC}"
}

print_step() {
    echo -e "${GREEN}  ┌─ $1${NC}"
}

print_success() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}  ⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}  ✗ $1${NC}"
}

# ======================== Pre-flight Checks ========================

print_header "Pre-flight Checks"

print_step "Checking Azure CLI installation"
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed"
    echo "  Install from: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

AZURE_VERSION=$(az version --output json | jq -r '."azure-cli"')
print_success "Azure CLI is installed (version: ${AZURE_VERSION})"

print_step "Checking Azure login status"
if ! az account show &> /dev/null; then
    print_error "Not logged into Azure"
    echo "  Run: az login"
    exit 1
fi

CURRENT_ACCOUNT=$(az account show --output json)
ACCOUNT_NAME=$(echo "$CURRENT_ACCOUNT" | jq -r '.user.name')
CURRENT_SUB=$(echo "$CURRENT_ACCOUNT" | jq -r '.id')
SUB_NAME=$(echo "$CURRENT_ACCOUNT" | jq -r '.name')

print_success "Logged in as: ${ACCOUNT_NAME}"
print_success "Subscription: ${SUB_NAME} (${CURRENT_SUB})"

if [ -z "$SUBSCRIPTION_ID" ]; then
    SUBSCRIPTION_ID="$CURRENT_SUB"
fi

# ======================== Display Configuration ========================

print_header "Configuration"

echo -e "${GREEN}  Resource Group:${NC}     ${RESOURCE_GROUP_NAME}"
echo -e "${GREEN}  Location:${NC}          ${LOCATION}"
echo -e "${GREEN}  ML Workspace:${NC}      ${WORKSPACE_NAME}"
echo -e "${GREEN}  Storage Account:${NC}   ${STORAGE_ACCOUNT_NAME}"
echo -e "${GREEN}  Container Registry:${NC} ${CONTAINER_REGISTRY_NAME}"
echo -e "${GREEN}  Subscription:${NC}      ${SUBSCRIPTION_ID}"

# ======================== Create Resource Group ========================

print_header "Creating Resource Group"

print_step "Creating resource group: ${RESOURCE_GROUP_NAME}"
if az group create \
    --name "$RESOURCE_GROUP_NAME" \
    --location "$LOCATION" \
    --output none; then
    print_success "Resource group created"
else
    print_error "Failed to create resource group"
    exit 1
fi

# ======================== Create Storage Account ========================

print_header "Creating Storage Account"

print_step "Creating storage account: ${STORAGE_ACCOUNT_NAME}"
if az storage account create \
    --name "$STORAGE_ACCOUNT_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --output none; then
    print_success "Storage account created"
else
    print_error "Failed to create storage account"
    exit 1
fi

print_step "Creating blob containers"
STORAGE_KEY=$(az storage account keys list \
    --account-name "$STORAGE_ACCOUNT_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --output json | jq -r '.[0].value')

az storage container create \
    --account-name "$STORAGE_ACCOUNT_NAME" \
    --name models \
    --account-key "$STORAGE_KEY" \
    --output none

az storage container create \
    --account-name "$STORAGE_ACCOUNT_NAME" \
    --name logs \
    --account-key "$STORAGE_KEY" \
    --output none

print_success "Blob containers created (models, logs)"

# ======================== Create Azure ML Workspace ========================

print_header "Creating Azure ML Workspace"

print_step "Creating ML workspace: ${WORKSPACE_NAME}"
if az ml workspace create \
    --name "$WORKSPACE_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --location "$LOCATION" \
    --storage-account "$STORAGE_ACCOUNT_NAME" \
    --output none 2>/dev/null; then
    print_success "ML workspace created"
    
    WORKSPACE_ID=$(az ml workspace show \
        --name "$WORKSPACE_NAME" \
        --resource-group "$RESOURCE_GROUP_NAME" \
        --output json | jq -r '.id')
    print_success "Workspace ID: ${WORKSPACE_ID}"
else
    print_warning "Azure ML workspace creation skipped (requires at least Contributor role)"
fi

# ======================== Create Container Registry ========================

print_header "Creating Container Registry"

print_step "Creating container registry: ${CONTAINER_REGISTRY_NAME}"
if az acr create \
    --name "$CONTAINER_REGISTRY_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --sku Basic \
    --admin-enabled true \
    --output none; then
    print_success "Container registry created"
    
    REGISTRY_URL=$(az acr show \
        --name "$CONTAINER_REGISTRY_NAME" \
        --resource-group "$RESOURCE_GROUP_NAME" \
        --query loginServer \
        --output tsv)
    
    REGISTRY_USER=$(az acr credential show \
        --name "$CONTAINER_REGISTRY_NAME" \
        --resource-group "$RESOURCE_GROUP_NAME" \
        --output json | jq -r '.username')
    
    REGISTRY_PASS=$(az acr credential show \
        --name "$CONTAINER_REGISTRY_NAME" \
        --resource-group "$RESOURCE_GROUP_NAME" \
        --output json | jq -r '.passwords[0].value')
    
    print_success "Registry URL: ${REGISTRY_URL}"
    print_success "Registry Username: ${REGISTRY_USER}"
else
    print_error "Failed to create container registry"
fi

# ======================== Create Key Vault ========================

print_header "Creating Key Vault"

KEY_VAULT_NAME="adas-keyvault-${RANDOM_SUFFIX}"

print_step "Creating Key Vault: ${KEY_VAULT_NAME}"
if az keyvault create \
    --name "$KEY_VAULT_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --location "$LOCATION" \
    --output none 2>/dev/null; then
    print_success "Key Vault created"
    
    print_step "Storing secrets"
    
    STORAGE_CONN_STR=$(az storage account show-connection-string \
        --name "$STORAGE_ACCOUNT_NAME" \
        --resource-group "$RESOURCE_GROUP_NAME" \
        --output tsv)
    
    az keyvault secret set \
        --vault-name "$KEY_VAULT_NAME" \
        --name "storage-connection-string" \
        --value "$STORAGE_CONN_STR" \
        --output none
    
    print_success "Stored storage connection string in Key Vault"
else
    print_warning "Key Vault creation skipped"
fi

# ======================== Summary ========================

print_header "Setup Summary"

print_success "✅ Resource Group: ${RESOURCE_GROUP_NAME}"
print_success "✅ Storage Account: ${STORAGE_ACCOUNT_NAME}"
print_success "✅ Container Registry: ${CONTAINER_REGISTRY_NAME}"
print_success "✅ ML Workspace: ${WORKSPACE_NAME}"
print_success "✅ Key Vault: ${KEY_VAULT_NAME}"

echo -e "\n${CYAN}📝 Next Steps:${NC}"

echo -e "\n${GREEN}1. BUILD AND PUSH DOCKER IMAGE:${NC}"
echo "   cd <project-directory>"
echo "   az acr build --registry ${CONTAINER_REGISTRY_NAME} --image adas-api:latest ."

echo -e "\n${GREEN}2. DEPLOY TO AZURE CONTAINER INSTANCES:${NC}"
echo "   az container create \\"
echo "     --resource-group ${RESOURCE_GROUP_NAME} \\"
echo "     --name ${CONTAINER_INSTANCE_NAME} \\"
echo "     --image ${REGISTRY_URL}/adas-api:latest \\"
echo "     --registry-username ${REGISTRY_USER} \\"
echo "     --registry-password <REGISTRY_PASSWORD> \\"
echo "     --ports 8000 8501 \\"
echo "     --environment-variables AZURE_ML_WORKSPACE=${WORKSPACE_NAME} \\"
echo "     --cpu 2 --memory 4"

echo -e "\n${GREEN}3. VIEW DEPLOYMENT:${NC}"
echo "   az container show --resource-group ${RESOURCE_GROUP_NAME} --name ${CONTAINER_INSTANCE_NAME}"

echo -e "\n${GREEN}4. VIEW LOGS:${NC}"
echo "   az container logs --resource-group ${RESOURCE_GROUP_NAME} --name ${CONTAINER_INSTANCE_NAME}"

echo -e "\n${GREEN}5. DELETE RESOURCES (when done):${NC}"
echo "   az group delete --name ${RESOURCE_GROUP_NAME} --yes --no-wait"

echo -e "\n${CYAN}📋 SAVE THESE VALUES FOR LATER:${NC}"
echo "   Subscription:        ${SUBSCRIPTION_ID}"
echo "   Resource Group:      ${RESOURCE_GROUP_NAME}"
echo "   Storage Account:     ${STORAGE_ACCOUNT_NAME}"
echo "   Container Registry:  ${CONTAINER_REGISTRY_NAME}"
echo "   Registry URL:        ${REGISTRY_URL}"
echo "   Registry Username:   ${REGISTRY_USER}"
echo "   ML Workspace:        ${WORKSPACE_NAME}"
echo "   Key Vault:           ${KEY_VAULT_NAME}"

echo -e "\n${GREEN}✅ Setup complete!${NC}\n"
