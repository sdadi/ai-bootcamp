// Only allow access to patient records for a physician in the same department.
// Requires a CUSTOM_JWT (OAuth) gateway whose tokens carry "role" and
// "department" claims, and a permit policy for the tool.
forbid (
    principal is AgentCore::OAuthUser,
    action == AgentCore::Action::"PatientRecords___get_patient_record",
    resource == AgentCore::Gateway::"<GATEWAY_ARN>"
)
unless {
    principal.hasTag("role") &&
    principal.getTag("role") == "physician" &&
    principal.hasTag("department") &&
    context.input has department &&
    principal.getTag("department") == context.input.department
};
