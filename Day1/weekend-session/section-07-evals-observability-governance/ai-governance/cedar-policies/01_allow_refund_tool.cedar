// Permit calls to the refund tool through this gateway.
// AgentCore Policy is default-deny: without a permit, every call is blocked.
permit (
    principal,
    action == AgentCore::Action::"RefundTarget___process_refund",
    resource == AgentCore::Gateway::"<GATEWAY_ARN>"
);
