from sklearn_f_para import Inv_para_prediction

if __name__ == "__main__":
    run = Inv_para_prediction(X_filename="snr00withoutnoise_new.csv", y_filename="snr00withoutnoise_inv_para_new.csv")
    run()