from datetime import date

def Raw2Dict(raw, predictions):
    today = date.today()
    return {
        "title": raw["input-title"],
        "content": raw["input-content"],
        "source": raw["input-src"],
        "published": FormatDate(raw["input-date"]),
        "analyzed": today,
        "model1": predictions[0],
        "model2": predictions[1],
        "model3": predictions[2]
    }
    
def FormatDate(date):
    date = date.split("/")
    return "{}-{}-{}".format(date[2], date[1], date[0])