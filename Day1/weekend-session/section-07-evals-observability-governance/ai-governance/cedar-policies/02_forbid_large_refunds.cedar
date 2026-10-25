// Deny refunds exceeding $1,000.
// Forbid wins over permit, so this overrides 01_allow_refund_tool.cedar
// whenever the amount is too large.
forbid (
    principal,
    action == AgentCore::Action::"RefundTarget___process_refund",
    resource == AgentCore::Gateway::"<GATEWAY_ARN>"
)
when {
    context.input.amount > 1000
};
