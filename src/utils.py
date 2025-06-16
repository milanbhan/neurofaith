from thefuzz import fuzz

def get_prediction_status(answers:list[str],
                        predictions:list[str],
                        threshold:int = 70)
    # Compute fuzzy similarity row by row
    result = answers.combine(predictions, lambda a, b: fuzz.ratio(a, b) >= threshold)

    return(result)