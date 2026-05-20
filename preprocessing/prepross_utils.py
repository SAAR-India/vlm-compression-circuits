def question_for_split(split_name: str, row: dict) -> str:
    """Generate VQA question per Visual-Counterfact split."""
    obj = row.get("object", "object")
    if split_name == "color":
        return f"What color is the {obj}?"
    return "Which object appears larger in the image?"