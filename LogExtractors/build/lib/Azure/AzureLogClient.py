import adal
import requests

def createServicePrincipal ():
    pass

def getADToken (tenantId, clientId, clientSecret):
    RESOURCE = "https://management.core.windows.net/"

    authority_url = "https://login.microsoftonline.com" + '/' + tenantId

    context = adal.AuthenticationContext(
        authority_url, validate_authority=tenantId != 'adfs',
        )

    token = context.acquire_token_with_client_credentials(
        RESOURCE,
        clientId,
        clientSecret)

    return token

class AzureLogClient (tenantId, clientId, clientSecret, subscription_id, resource_group, app_insights_name, endpointName):