from thefuzz import fuzz

def get_prediction_status(answers:list[str],
                        predictions:list[str],
                        threshold:int = 70) -> list:
    # Compute fuzzy similarity row by row
    fuzzy_result = answers.combine(predictions, lambda a, b: fuzz.ratio(a, b) >= threshold)
    contain_result = [answer in prediction for answer, prediction in zip(answers, predictions)]
    result = [fuzzy or contain for fuzzy, contain in zip(fuzzy_result, contain_result)]

    return(result)

def get_explanation_status(bridge_objects:list[str],
                        explanations:list[str],
                        predicted_bridge_objects:list[str],
                        threshold:int = 50) -> list:
    # Compute fuzzy similarity row by row
    fuzzy_result = bridge_objects.reset_index(drop=True).combine(predicted_bridge_objects.reset_index(drop=True), lambda a, b: fuzz.ratio(a, b) >= threshold)
    contain_result = [bridge_object in explanation for bridge_object, explanation in zip(bridge_objects.fillna(" "), explanations.fillna(" "))]
    result = [fuzzy or contain for fuzzy, contain in zip(fuzzy_result, contain_result)]

    return(result)


def clean_prediction(explanations:list[str]) -> list[str]:
    explanations = explanations.str.split('\nThis statement is').str[0].str.strip()
    explanations = explanations.str.split('\nPlease provide more context or').str[0].str.strip()
    explanations = explanations.str.replace("\n","")
    explanations = explanations.str.replace(":","")
    explanations = explanations.str.replace("?","")
    explanations = explanations.str.replace("!","")
    explanations = explanations.str.replace("named","")
    explanations = explanations.str.split('Let me know if').str[0].str.strip()
    explanations = explanations.str.split(".Here's").str[0].str.strip()
    explanations = explanations.str.rstrip()
    return(explanations)

def clean_explanation(explanations:list[str]) -> list[str]:
    explanations = explanations.str.split('Let me know').str[0].str.strip()
    explanations = explanations.str.split('Please give').str[0].str.strip()
    explanations = explanations.str.replace("\n","")
    explanations = explanations.str.rstrip()
    return(explanations)

def clean_interpretation(explanations:list[str]) -> list[str]:
    explanations = explanations.str.split('Let me know').str[0].str.strip()
    explanations = explanations.str.split('Please give').str[0].str.strip()
    explanations = explanations.str.replace("**Answer:**","")
    explanations = explanations.str.replace("*Answer*","")
    explanations = explanations.str.replace("Answer: ","")
    explanations = explanations.str.replace("What is the following?"," ")
    explanations = explanations.str.replace('**"Answer"**'," ")
    explanations = explanations.str.replace("\n"," ")
    explanations = explanations.str.replace(r'\s+', ' ', regex=True)
    
    explanations = explanations.str.rstrip()
    return(explanations)

def clean_bridge_objects(bridge_objects:list[str]) -> list[str]:
    bridge_objects = bridge_objects.str.split('Let me know').str[0].str.strip()
    bridge_objects = bridge_objects.str.split('Please give').str[0].str.strip()
    bridge_objects = bridge_objects.str.replace("|im_end|","")
    bridge_objects = bridge_objects.str.replace("<|endoftext|>","")
    bridge_objects = bridge_objects.str.replace("*","")
    bridge_objects = bridge_objects.str.rstrip()
    return(bridge_objects)