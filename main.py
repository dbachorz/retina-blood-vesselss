from random import randint
import cv2
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, accuracy_score
from imblearn.metrics import geometric_mean_score
from math import sqrt

offset = 20
samples_number = 200

def learn(eye_type):
    prefix = "h" if eye_type == "healthy" else "dr"
    # obraz na ktorym bedzie odbywalo sie trenowanie klasyfikatora
    training_image = cv2.imread(eye_type + "/01_" + prefix + ".jpg")
    # wycinek 700x700, na tym wycinku klasyfikator sie uczy
    training_image = training_image[700:1400,700:1400,:]

    training_expert_mask = cv2.imread(eye_type + "_manualsegm/01_" + prefix + ".tif")
    training_expert_mask = training_expert_mask[700:1400,700:1400,:]

    train_x = np.zeros((samples_number, (2*offset)**2))
    train_y = np.zeros((samples_number))

    # przygotowanie zbioru uczącego
    # losowanie punktu na naszym wycinku i otaczanie go zadanym offsetem
    for i in range(0,samples_number):
        x = randint(0+offset, training_image.shape[0]-offset)
        y = randint(0+offset, training_image.shape[1]-offset)
        crop_img = training_image[x-offset:x+offset, y-offset:y+offset]
        flattened = crop_img[:,:,1].flatten()
        train_x[i,:] = flattened
        train_y[i] = training_expert_mask[x,y,1]/255
        
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(offset, 2), random_state=1)

    # train_x to macierz [zawartość próbki] x [liczba próbek]
    # train_y to wektor zawierający inf czy piksel w masce obrazu eksperta to tło czy naczynie
    clf.fit(train_x, train_y)
    return clf

def predict(clf, img_num, eye_type):
    prefix = "h" if eye_type == "healthy" else "dr"
    # predykcja na calym obrazie
    img_to_be_processed = cv2.imread(eye_type + "/" + img_num + "_" + prefix + ".jpg")
    expert_result = cv2.imread(eye_type + "_manualsegm/" + img_num + "_" + prefix + ".tif")

    predict_x = np.zeros(((img_to_be_processed.shape[1]-2*offset), (2*offset)**2))

    res = np.zeros(((img_to_be_processed.shape[0]-2*offset) * (img_to_be_processed.shape[1]-2*offset)))
    for i in range(offset, img_to_be_processed.shape[0]-offset):
        a = 0
        for j in range(offset, img_to_be_processed.shape[1]-offset):
            predict_crop_img = img_to_be_processed[i-offset:i+offset, j-offset:j+offset]
            predict_x[a,:] = predict_crop_img[:,:,1].flatten()
            a=a+1
        res[(i-offset) * (img_to_be_processed.shape[1]-2*offset) : (i-offset) * (img_to_be_processed.shape[1]-2*offset)+a] = clf.predict(predict_x)

    learned_result = res.reshape((img_to_be_processed.shape[0]-2*offset, img_to_be_processed.shape[1]-2*offset))
    learned_result = learned_result*255
    expert_result = expert_result[:,:,1] # wybieram tylko 1 kanal (zawsze w czarno bialym obrazie wszystkie 3 sa takie same)

    # przygotowanie danych wejsciowych do raportu z trafnosciami
    # maska do testow
    fovmask = cv2.imread(eye_type + "_fovmask/" + img_num + "_" + prefix + "_mask.tif")
    fovmask = fovmask[:,:,1] # wybieram tylko 1 kanal (zawsze w czarno bialym obrazie wszystkie 3 sa takie same)

    # rysowanie obrazow
    # cv2.imshow("cropped", learned_result)
    # cv2.imshow("cropped2", expert_result)
    # cv2.imshow("fovmask", fovmask)
    learned_result_with_offset = np.zeros(fovmask.shape); # na poczatku wypelniam macierz o rozmiarze maski zerami
    learned_result_with_offset[offset: -offset,offset: -offset] = learned_result # uzupelniam nasz wynik o brakujace offsety

    # obraz z nalozona maska
    learned_result_with_mask = learned_result_with_offset[fovmask==255]
    expert_result_with_mask = expert_result[fovmask==255]

    # spłaszczanie rezultatów w celu przygotowania na wejście funkcji liczących trafności
    learned_result_with_mask = learned_result_with_mask.flatten()
    expert_result_with_mask = expert_result_with_mask.flatten()

    recalled_vessel_score = recall_score(expert_result_with_mask, learned_result_with_mask, average='macro')
    recalled_background_score = recall_score(255 - expert_result_with_mask, 255 - learned_result_with_mask, average='macro')
    accured_score = accuracy_score(expert_result_with_mask, learned_result_with_mask)
    geometric_avg_score = geometric_mean_score(expert_result_with_mask, learned_result_with_mask)

    print (img_num + " recalled_score", recalled_vessel_score)
    print (img_num + " accured_score", accured_score)
    print (img_num + " recalled_background_score", recalled_background_score)
    print (img_num + " geometric_avg_score", geometric_avg_score)

    return recalled_vessel_score, accured_score, recalled_background_score, geometric_avg_score

    # cv2.waitKey(0) aby oczekiwać na dowolny klawisz po wyświetleniu zdjęć
    
print ("start uczenie")
clf = learn("healthy") # healthy- zdrowe, diabetic_retinopathy- chore

recalled_vessel_scores = 0
accured_scores = 0
recalled_background_scores = 0
geometric_avg_scores = 0

file_nums_arr = ["02"]

print ("start przewidywanie dla zdrowego")
# dla zdrowego
for file_num in file_nums_arr:
    recalled_vessel_score, accured_score, recalled_background_score, geometric_avg_score = predict(clf, file_num, "healthy")
    recalled_vessel_scores += recalled_vessel_score
    accured_scores += accured_score
    recalled_background_scores += recalled_background_score
    geometric_avg_scores += geometric_avg_score

print ("================= OSTATECZNE WYNIKI DLA OKA ZDROWEGO =================")
print ("srednia trafnosc naczynie: ", recalled_vessel_scores / len(file_nums_arr))
print ("srednia ogolna trafnosc: ", accured_scores / len(file_nums_arr))
print ("srednia trafnosc tlo: ", recalled_background_scores / len(file_nums_arr))
print ("srednia srednia geometryczna: ", geometric_avg_scores / len(file_nums_arr))

recalled_vessel_scores = 0
accured_scores = 0
recalled_background_scores = 0
geometric_avg_scores = 0

print ("start przewidywanie dla chorego")
# dla chorego
for file_num in file_nums_arr:
    recalled_vessel_score, accured_score, recalled_background_score, geometric_avg_score = predict(clf, file_num, "diabetic_retinopathy")
    recalled_vessel_scores += recalled_vessel_score
    accured_scores += accured_score
    recalled_background_scores += recalled_background_score
    geometric_avg_scores += geometric_avg_score

print ("================= OSTATECZNE WYNIKI DLA OKA CHOREGO =================")
print ("srednia trafnosc naczynie: ", recalled_vessel_scores / len(file_nums_arr))
print ("srednia ogolna trafnosc: ", accured_scores / len(file_nums_arr))
print ("srednia trafnosc tlo: ", recalled_background_scores / len(file_nums_arr))
print ("srednia srednia geometryczna: ", geometric_avg_scores / len(file_nums_arr))
