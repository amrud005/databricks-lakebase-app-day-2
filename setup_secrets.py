"""
One-time setup script: stores the Lakebase connection URL in a Databricks
secret scope. Run from a notebook or local CLI — never commit the secret.

Usage:
    python setup_secrets.py
"""
import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

w = WorkspaceClient()

# Uncomment once if the scope does not exist yet:
# w.secrets.create_scope(scope="database")

w.secrets.put_secret(
    scope="database",
    key="lakebase-url",
    string_value=getpass.getpass("Paste your Lakebase URL: "),
)

w.secrets.put_acl(
    scope="database",
    principal="users",
    permission=workspace.AclPermission.READ,
)

print("Stored secret database/lakebase-url")
