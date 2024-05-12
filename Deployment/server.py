import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.datastructures import FileStorage
from Ecg import ECG

app = FastAPI()

# Initialize ECG object
ecg = ECG()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["*"],
# )

print("MERA NAAM HAI HAI")
@app.post("/predict/")
async def predict_ecg(file: UploadFile = File(...)):
    # Get the uploaded image
    contents = FileStorage(
        stream=file.file,
        filename=file.filename,
        content_type=file.content_type,
        # content_length=file.file._length,
    )
    print(contents)
    # Convert contents to bytes if it's a string
    # if isinstance(contents, str):
    #     contents = contents.encode()

    # Call the necessary methods from your ECG class to process the image and make a prediction
    ecg_user_image_read = ecg.getImage(contents)
    print("YES ")
    ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
    dividing_leads=ecg.DividingLeads(ecg_user_image_read)
    ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)
    ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
    ecg_1dsignal = ecg.CombineConvert1Dsignal()
    ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
    prediction = ecg.ModelLoad_predict(ecg_final)

    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
