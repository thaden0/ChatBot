from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_401_UNAUTHORIZED
from fastapi import Header

bearer = HTTPBearer(auto_error=False)

def require_bearer(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    if not creds or creds.scheme.lower() != "bearer" or creds.credentials != "EXPECTED_TOKEN":
        raise HTTPException(HTTP_401_UNAUTHORIZED, "Invalid token")
    return True

async def require_api_key(x_api_key: str = Header(None)):
    if x_api_key != "EXPECTED_API_KEY":
        raise HTTPException(HTTP_401_UNAUTHORIZED, "Invalid API key")
    return True
