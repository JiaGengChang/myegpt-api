import re

# present structured AI response or unstructured tool response in html
def format_text_message(message) -> str:
    if isinstance(message, dict):
        content = message.get('text', '')
    else:
        content = str(message)
    
    # Detect if the content contains tool results (e.g., text within [ ])
    if re.search(r"\[\(.*?\)\]", content):
        formatted = f"Tool result: {content}"
    else:
        # content is a text response
        formatted = f"Agent message: {content}"

    return formatted

def format_tool_message(message) -> str:

    if isinstance(message, dict):
        tool_result = message.get('content')
        tool_name = message.get('name')
        formatted = f"`{tool_name}` tool call result: {tool_result}"
    else:
        formatted = str(message)

    return formatted