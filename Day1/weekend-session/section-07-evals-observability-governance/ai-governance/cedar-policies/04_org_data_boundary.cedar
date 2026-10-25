// Restrict data access to the caller's own organisation (multi-tenant boundary).
// The org comes from the authenticated token, not from the prompt, so a prompt
// injection cannot query another tenant's data.
forbid (
    principal is AgentCore::OAuthUser,
    action == AgentCore::Action::"DataAPI___query_database",
    resource == AgentCore::Gateway::"<GATEWAY_ARN>"
)
unless {
    principal.hasTag("org_id") &&
    context.input has org_id &&
    principal.getTag("org_id") == context.input.org_id
};
