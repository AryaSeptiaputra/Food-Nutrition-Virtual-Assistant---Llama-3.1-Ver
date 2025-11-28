def get_system_prompt(tools_desc: str, tool_names: list, template_path: str) -> str:
    """Read system prompt template from disk and inject tool information.
    
    Loads a prompt template file and replaces placeholders with actual tool names
    and descriptions. If template file is not found, returns a default fallback prompt.
    Supports two placeholders: {tool_names} for comma-separated tool names and 
    {tools} for the detailed tool descriptions.
    
    Args:
        tools_desc: Formatted string describing all available tools and their usage
        tool_names: List of tool names to inject into the {tool_names} placeholder
        template_path: File path to the system prompt template
    
    Returns:
        Complete system prompt string with injected tool information. 
        Falls back to default prompt if template file not found.
    
    Note:
        - Prints confirmation message when tool names are successfully injected
        - Gracefully handles missing template file by returning a default prompt
    """

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            system_prompt_text = f.read()
    except FileNotFoundError:
        return f"You are an AI assistant. Available tools: {tools_desc}"

    if tool_names:
        tool_names_str = ", ".join(tool_names)
        system_prompt_text = system_prompt_text.replace("{tool_names}", tool_names_str)
        print(f"✅ Tool names injected: {tool_names_str}")

    if "{tools}" in system_prompt_text:
        system_prompt_text = system_prompt_text.replace("{tools}", tools_desc)

    return system_prompt_text