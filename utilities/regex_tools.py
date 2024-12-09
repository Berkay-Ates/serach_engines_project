def apply_custom_regex(value, compiled_regex):
    return value.str.replace(compiled_regex, "", regex=True)
