import pickle

#doc_new = ['obama is running for president in 2016']

var = input("Please enter the news text you want to verify: ")
print("You entered: " + str(var))
model_number = input("Please enter the model name to infer: 1.LR, 2.RF, 3.XGB, 4.NB ")
print("You entered: " + str(model_number))

#function to run for prediction
def detecting_fake_news(var,model_number):    
#retrieving the best model for prediction call
    switcher = {
        1: "lr.bin",
        2: "rf.bin",
        3: "xgb.bin",
        4: "nb.bin"
    }
    model_name = switcher.get(argument, "Invalid model number")

    model_name= 'final_model.sav'
    load_model = pickle.load(open(model_name, 'rb'))
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    return (print("The given statement is ",prediction[0]),
        print("The truth probability score is ",prob[0][1]))


if __name__ == '__main__':
    detecting_fake_news(var,model_number)