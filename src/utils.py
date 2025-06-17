from thefuzz import fuzz

def get_prediction_status(answers:list[str],
                        predictions:list[str],
                        threshold:int = 70) -> list:
    # Compute fuzzy similarity row by row
    fuzzy_result = answers.combine(predictions, lambda a, b: fuzz.ratio(a, b) >= threshold)
    contain_result = [answer in prediction for answer, prediction in zip(answers, predictions)]
    result = [fuzzy or contain for fuzzy, contain in zip(fuzzy_result, contain_result)]

    return(result)

def clean_explanation(explanations:list[str]) -> list[str]:
    explanations = explanations.str.split('\nThis statement is').str[0].str.strip()
    explanations = explanations.str.split('\nPlease provide more context or').str[0].str.strip()
    explanations = explanations.str.replace("\n","")
    explanations = explanations.str.split('Let me know if').str[0].str.strip()
    explanations = explanations.str.rstrip()
    return(explanations)