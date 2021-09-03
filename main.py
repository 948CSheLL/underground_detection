import math
from sklearn_f_para import Inv_para_prediction

if __name__ == "__main__":
    # run = Inv_para_prediction()
    run = Inv_para_prediction(X_filename="datas/snr05.csv", y_filename="datas/snr05_inv_para.csv")
    # run = Inv_para_prediction(X_filename="datas/snr10.csv", y_filename="datas/snr10_inv_para.csv")
    # run = Inv_para_prediction(X_filename="datas/snr15.csv", y_filename="datas/snr15_inv_para.csv")
    # run = Inv_para_prediction(X_filename="datas/snr20.csv", y_filename="datas/snr20_inv_para.csv")
    # run = Inv_para_prediction(X_filename="datas/snr25.csv", y_filename="datas/snr25_inv_para.csv")
    # run = Inv_para_prediction(X_filename="datas/snr30.csv", y_filename="datas/snr30_inv_para.csv")
    # run = Inv_para_prediction(X_filename="datas/snr00withoutnoise_new.csv", y_filename="datas/snr00withoutnoise_inv_para_new.csv")
    run()