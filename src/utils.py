from thefuzz import fuzz

def get_prediction_status(answers:list[str],
                        predictions:list[str],
                        threshold:int = 70) -> list:
    # Compute fuzzy similarity row by row
    result = answers.combine(predictions, lambda a, b: fuzz.ratio(a, b) >= threshold)

    return(result)

def clean_explanation(explanations:list[str]) -> list[str]:
    explanations = explanations.str.replace("\n","")
    explanations = explanations.str.split('Let me know if').str[0].str.strip()
    explanations = explanations.str.rstrip()