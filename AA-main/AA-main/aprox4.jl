using FileIO
using Images
using Plots
using StatsBase
using StatsPlots
include("funcs.jl")
include("own_funcs.jl")

seed!(1);



# # ----------------------------------------------------------------------------------------------------------
# # Load dataset
# # ----------------------------------------------------------------------------------------------------------

#######################################################################################
# Ruta de la carpeta del dataset de imágenes relativa a la ubicación main.jl
dir = "./dataset/train/";
test_dir = "./dataset/test/";
#######################################################################################

#######################################################################################
# Terminaciones de los archivos de dataset que clasifican las imágenes
# en el numero de dedos levantados y mano izquierda y derecha
endingsR = [string(i, "R") for i in 0:5];
endingsL = [string(i, "L") for i in 0:5];
#######################################################################################


#######################################################################################
# Se cargan la imágenes del dataset en un primer vector donde cada elemento 
# se organiza por el numero de dedos levantados
# es un vector con indices 1:6
# donde 1 corresponde a 0 dedos levantados, 2 a 1 dedo levantado, etc.
# 
# A su vez cada elemento del primer vector, es un segundo vector
# con todas las imágenes del dataset del indice que corresponde
# al numero de dedos levantados.
# Y una imagen es un Array de dos dimensiones donde cada elemento es un pixel.
imgsRarr = [loadImages(dir; ending=endingR) for endingR in endingsR];
imgsLarr = [loadImages(dir; ending=endingL) for endingL in endingsL];

test_imgsRarr = [loadImages(test_dir; ending=endingR) for endingR in endingsR];
test_imgsLarr = [loadImages(test_dir; ending=endingL) for endingL in endingsL];
# imgsR serían todas las imágenes del dataset de la mano derecha, un vector.
# imgsR[1] serían todas las imágenes de 0 dedos levantadas, también vector.
# imgsR[1][1] sería la primera imagen de 0 dedos levantados, un array de dos dimensiones.
#######################################################################################

#######################################################################################
# Se establece que clases del dataset se usarán (0, 1 dedos, etc.)
aprox_index = [0,1,2,3,4,5];
# Todas las imágenes en un mismo vector
imgsR = cat_vector_of_imgs(imgsRarr, aprox_index);
imgsL = cat_vector_of_imgs(imgsLarr, aprox_index);

imgs = vcat(imgsR, imgsL);

test_imgsR = cat_vector_of_imgs(test_imgsRarr, aprox_index);
test_imgsL = cat_vector_of_imgs(test_imgsLarr, aprox_index);

test_imgs = vcat(test_imgsR, test_imgsL);
#######################################################################################


#######################################################################################
# Se genera un vector donde cada elemento sera el número de dedos levantados
# correspondiente a cada instancia
targets_R = gen_targets(imgsRarr, aprox_index);
targets_L = gen_targets(imgsLarr, aprox_index);

test_targets_R = gen_targets(test_imgsRarr, aprox_index);
test_targets_L = gen_targets(test_imgsLarr, aprox_index);
#######################################################################################


#######################################################################################
# Se concatenan las matrices de targets de mano derecha e izquierda
# para tenerlas en una única matriz.
targets =  vcat(targets_R, targets_L);

test_targets =  vcat(test_targets_R, test_targets_L);
#######################################################################################




# # ----------------------------------------------------------------------------------------------------------
# # Preprocessing dataset
# # ----------------------------------------------------------------------------------------------------------

#######################################################################################
# Se convierten a imágenes binarias dando un umbral para la luminosidad/luminancia.
# Pasado el umbral será 1/true si no 0/false, el resultado es la imagen en una BitMatrix.
# Todas las imágenes binarias en un vector, para extraer características.
# Correspondencia de indices con la BitMatrix de targets.
bool_imgs = imageToBoolM.(imgs; threshold=0.4);

test_bool_imgs = imageToBoolM.(test_imgs; threshold=0.4);
#######################################################################################


#######################################################################################
# La siguiente función transforma un array booleano (imagen umbralizada) en un array de etiquetas
# Cada grupo de píxeles puesto como "true" en la matriz booleana y conectados se le asigna una etiqueta
labelsArraysImgs = ImageMorphology.label_components.(bool_imgs);

test_labelsArraysImgs = ImageMorphology.label_components.(test_bool_imgs);
#######################################################################################

#######################################################################################
# Se calculan los tamaños de los elementos de cada label array
sizes_labelArraysImgs = ImageMorphology.component_lengths.(labelsArraysImgs);
test_sizes_labelArraysImgs = ImageMorphology.component_lengths.(test_labelsArraysImgs);
#######################################################################################


#######################################################################################
# Se eliminan elementos de los label array un tamaño mínimo
labels_to_remove = get_labels_min_size_to_remove(sizes_labelArraysImgs);
labelsArraysImgs = remove_labels_from_labelsArrays(labels_to_remove, labelsArraysImgs);

test_labels_to_remove = get_labels_min_size_to_remove(test_sizes_labelArraysImgs);
test_labelsArraysImgs = remove_labels_from_labelsArrays(test_labels_to_remove, test_labelsArraysImgs);
# Se eliminan los elementos del sizes_labelArraysImgs
sizes_labelArraysImgs = map((sizes_labelArraysImgs, labels_to_remove) -> sizes_labelArraysImgs[setdiff(begin:end, labels_to_remove)] ,sizes_labelArraysImgs, labels_to_remove);

test_sizes_labelArraysImgs = map((test_sizes_labelArraysImgs, test_labels_to_remove) -> test_sizes_labelArraysImgs[setdiff(begin:end, test_labels_to_remove)] ,test_sizes_labelArraysImgs, test_labels_to_remove);
#######################################################################################


#######################################################################################
# Se calculan los bounding boxes
boundingBoxesImgs = ImageMorphology.component_boxes.(labelsArraysImgs);

test_boundingBoxesImgs = ImageMorphology.component_boxes.(test_labelsArraysImgs);
#######################################################################################


#######################################################################################
# Se ordenan los bounding boxes de mayor a menor para descartar objetos no deseados, tomando el
# segundo objeto de mayor tamaño que debería ser la mano (el primero es todo el fondo)

# Se calculan las posiciones en orden descendente de cada elemento, ej. a = [3, 0, 1, 2] => [1, 4, 3, 2]
indexes_sorted_sizes_labelArraysImgs = sortperm.(sizes_labelArraysImgs, rev=true)

test_indexes_sorted_sizes_labelArraysImgs = sortperm.(test_sizes_labelArraysImgs, rev=true)
# Se toma el numero de label que tendría la segunda posición (2) y se haya su indice en el vector
# Ej. a => 4, (el 2, el segundo elemento mas grande, esta en la posición 4)
labelsToRemain = findall.(x -> x == 2, indexes_sorted_sizes_labelArraysImgs)

test_labelsToRemain = findall.(x -> x == 2, test_indexes_sorted_sizes_labelArraysImgs)
# Se toma los bounding boxes de esa posición (i.-1 por ser un OffsetArray y empezar en 0)
boundingBoxesImgs = map((boundingBoxesImgs, i) -> boundingBoxesImgs[i.-1], boundingBoxesImgs, labelsToRemain)

test_boundingBoxesImgs = map((test_boundingBoxesImgs, i) -> test_boundingBoxesImgs[i.-1], test_boundingBoxesImgs, test_labelsToRemain)
#######################################################################################


#######################################################################################
# Se calculan las proporciones de alto y ancho del segundo bounding box
# que debería ser la mano, el primero con indice 0 es el del fondo.
inputs = Matrix{Float32}(undef,length(bool_imgs),0);
inputs = hcat(inputs, get_width_length_boxes_ratio(boundingBoxesImgs));

test_inputs = Matrix{Float32}(undef,length(test_bool_imgs),0);
test_inputs = hcat(test_inputs, get_width_length_boxes_ratio(test_boundingBoxesImgs));
#######################################################################################


#######################################################################################
# Se calcula la proporción de 1s y 0s de la imagen dentro de la box calculada
boxBoolImgs = get_box_img.(bool_imgs, boundingBoxesImgs);
inputs = hcat(inputs, get_ones_zeros_ratio(boxBoolImgs));

test_boxBoolImgs = get_box_img.(test_bool_imgs, test_boundingBoxesImgs);
test_inputs = hcat(test_inputs, get_ones_zeros_ratio(test_boxBoolImgs));
#######################################################################################



#######################################################################################
ratio = 5/16
inputs = hcat(inputs, get_count_horizontal_intersections(boxBoolImgs, ratio));
test_inputs = hcat(test_inputs, get_count_horizontal_intersections(test_boxBoolImgs, ratio));

#######################################################################################


#######################################################################################
# Métricas de las características

imgs_per_pattern = 1500;
n_aprox_index = length(aprox_index);

println("\nCaracterística 1 => proporciones de alto y ancho del bounding box de la mano")
println("Característica 2 => proporción de 1s y 0s de la imagen dentro de la box de la mano")
println("Característica 3 => números de agrupaciones de unos en la linea definida")

for i in axes(inputs,2)
  println();
  for c in (aprox_index)
    println("Media característica $i, clase $c: ", mean(vcat(inputs[c*imgs_per_pattern+1:(c+1)*imgs_per_pattern, i], inputs[(c+n_aprox_index)*imgs_per_pattern+1:(c+n_aprox_index+1)*imgs_per_pattern, i])))
    println("Desviación característica $i, clase $c: ", std(vcat(inputs[c*imgs_per_pattern+1:(c+1)*imgs_per_pattern, i], inputs[(c+n_aprox_index)*imgs_per_pattern+1:(c+n_aprox_index+1)*imgs_per_pattern, i])))
  end
end
println("\n")
#######################################################################################


inputs = normalizeMinMax(inputs);

# # ----------------------------------------------------------------------------------------------------------
# # Testing models
# # ----------------------------------------------------------------------------------------------------------

crossValidationIndices = crossvalidation(targets, 5);


topology = Int[3];
learningRate = Float32(0.01);
validationRatio = 0.25;
numExecutions = 20;
maxEpochs = 500;
maxEpochsVal = 20;


modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numExecutions;
modelHyperparameters["maxEpochs"] = maxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
println("============================================================")
println("ANN-1:")
ANN_results1 = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology2 = Int[3, 5];
modelHyperparameters["topology"] = topology2;
println("============================================================")
println("ANN-2:")
ANN_results2 = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology3 = Int[1, 2];
modelHyperparameters["topology"] = topology3;
println("============================================================")
println("ANN-3:")
ANN_results3 = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology4 = Int[4];
modelHyperparameters["topology"] = topology4;
println("============================================================")
println("ANN-4:")
ANN_results4 = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology5 = Int[5, 3];
modelHyperparameters["topology"] = topology5;
println("============================================================")
println("ANN-5:")
ANN_results5 = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology6 = Int[2, 2];
modelHyperparameters["topology"] = topology6;
println("============================================================")
println("ANN-6:")
ANN_results6 = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology7 = Int[6];
modelHyperparameters["topology"] = topology7;
println("============================================================")
println("ANN-7:")
ANN_results7 = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology8 = Int[4, 1];
modelHyperparameters["topology"] = topology8;
println("============================================================")
println("ANN-8:")
ANN_results8 = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);



kernel = "rbf";
degree = 3;
gamma = 2;
C=1;

modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["degree"] = degree;
modelHyperparameters["gamma"] = gamma;
modelHyperparameters["C"] = C;
println("============================================================")
println("SVM-1:")
SVM_results1 = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);


kernel2 = "rbf";
C2=0.5;
modelHyperparameters["kernel"] = kernel2;
modelHyperparameters["C"] = C2;
println("============================================================")
println("SVM-2:")
SVM_results2 = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

kernel3 = "rbf";
C3=2;
modelHyperparameters["kernel"] = kernel3;
modelHyperparameters["C"] = C3;
println("============================================================")
println("SVM-3:")
SVM_results3 = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

kernel4 = "linear";
C4=0.5;
modelHyperparameters["kernel"] = kernel4;
modelHyperparameters["C"] = C4;
println("============================================================")
println("SVM-4:")
SVM_results4 = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

kernel5 = "linear";
C5=1;
modelHyperparameters["kernel"] = kernel5;
modelHyperparameters["C"] = C5;
println("============================================================")
println("SVM-5:")
SVM_results5 = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

kernel6 = "linear";
C6=1.5;
modelHyperparameters["kernel"] = kernel6;
modelHyperparameters["C"] = C6;
println("============================================================")
println("SVM-6:")
SVM_results6 = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

kernel7 = "poly";
C7=1;
modelHyperparameters["kernel"] = kernel7;
modelHyperparameters["C"] = C7;
println("============================================================")
println("SVM-7:")
SVM_results7 = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

kernel8 = "poly";
C8=1.5;
modelHyperparameters["kernel"] = kernel8;
modelHyperparameters["C"] = C8;
println("============================================================")
println("SVM-8:")
SVM_results8 = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);



max_depth = 4;

modelHyperparameters = Dict();
modelHyperparameters["max_depth"] = max_depth;
println("============================================================")
println("DecisionTree-1:")
DecisionTree_results1 = modelCrossValidation(:DecisionTree, modelHyperparameters, inputs, targets, crossValidationIndices);

max_depth2 = 3;
modelHyperparameters["max_depth"] = max_depth2;
println("============================================================")
println("DecisionTree-2:")
DecisionTree_results2 = modelCrossValidation(:DecisionTree, modelHyperparameters, inputs, targets, crossValidationIndices);

max_depth3 = 5;
modelHyperparameters["max_depth"] = max_depth3;
println("============================================================")
println("DecisionTree-3:")
DecisionTree_results3 = modelCrossValidation(:DecisionTree, modelHyperparameters, inputs, targets, crossValidationIndices);

max_depth4 = 10;
modelHyperparameters["max_depth"] = max_depth4;
println("============================================================")
println("DecisionTree-4:")
DecisionTree_results4 = modelCrossValidation(:DecisionTree, modelHyperparameters, inputs, targets, crossValidationIndices);

max_depth5 = 8;
modelHyperparameters["max_depth"] = max_depth5;
println("============================================================")
println("DecisionTree-5:")
DecisionTree_results5 = modelCrossValidation(:DecisionTree, modelHyperparameters, inputs, targets, crossValidationIndices);

max_depth6 = 15;
modelHyperparameters["max_depth"] = max_depth6;
println("============================================================")
println("DecisionTree-6:")
DecisionTree_results6 = modelCrossValidation(:DecisionTree, modelHyperparameters, inputs, targets, crossValidationIndices);




n_neighbors = 3;

modelHyperparameters = Dict();
modelHyperparameters["n_neighbors"] = n_neighbors;
println("============================================================")
println("kNN-1:")
kNN_results1 = modelCrossValidation(:kNN, modelHyperparameters, inputs, targets, crossValidationIndices);

n_neighbors2 = 2;
modelHyperparameters["n_neighbors"] = n_neighbors2;
println("============================================================")
println("kNN-2:")
kNN_results2 = modelCrossValidation(:kNN, modelHyperparameters, inputs, targets, crossValidationIndices);

n_neighbors3 = 5;
modelHyperparameters["n_neighbors"] = n_neighbors3;
println("============================================================")
println("kNN-3:")
kNN_results3 = modelCrossValidation(:kNN, modelHyperparameters, inputs, targets, crossValidationIndices);

n_neighbors4 = 8;
modelHyperparameters["n_neighbors"] = n_neighbors4;
println("============================================================")
println("kNN-4:")
kNN_results4 = modelCrossValidation(:kNN, modelHyperparameters, inputs, targets, crossValidationIndices);

n_neighbors5 = 10;
modelHyperparameters["n_neighbors"] = n_neighbors5;
println("============================================================")
println("kNN-5:")
kNN_results5 = modelCrossValidation(:kNN, modelHyperparameters, inputs, targets, crossValidationIndices);

n_neighbors6 = 7;
modelHyperparameters["n_neighbors"] = n_neighbors6;
println("============================================================")
println("kNN-6:")
kNN_results6 = modelCrossValidation(:kNN, modelHyperparameters, inputs, targets, crossValidationIndices);


ANN_results1
ANN_results2
ANN_results3
ANN_results4
ANN_results5
ANN_results6
ANN_results7
ANN_results8

SVM_results1
SVM_results2
SVM_results3
SVM_results4
SVM_results5
SVM_results6
SVM_results7
SVM_results8

DecisionTree_results1
DecisionTree_results2
DecisionTree_results3
DecisionTree_results4
DecisionTree_results5
DecisionTree_results6

kNN_results1
kNN_results2
kNN_results3
kNN_results4
kNN_results5
kNN_results6




#######################################################################################
test_inputs = normalizeMinMax(test_inputs);
#######################################################################################

#######################################################################################
# Los modelos de cada tipo con mejor precisión se volverán a entrenar y se probarán con las imágenes de test
keys_model = ["acc", "errorRate", "recall", "specificity", "VPP", "VPN", "F1", "confMatrix"];
#######################################################################################

#######################################################################################
# Modelo árbol de decisión 6 (Tiene la mayor precisión entre todos los tipos)
model = DecisionTreeClassifier(max_depth=max_depth6, random_state=1);

model = fit!(model, inputs, targets);

testOutputs = predict(model, test_inputs);

results = printConfusionMatrix(testOutputs, test_targets);
DTC_test_results = Dict(zip(keys_model,collect(results)));
#######################################################################################

#######################################################################################
# Modelo kNN 2
model = KNeighborsClassifier(n_neighbors2);

model = fit!(model, inputs, targets);

testOutputs = predict(model, test_inputs);

results = printConfusionMatrix(testOutputs, test_targets);
kNN_test_results = Dict(zip(keys_model,collect(results)));
#######################################################################################

#######################################################################################
# Modelo SVM 3
model = SVC(kernel=kernel3, degree=degree, gamma=gamma, C=C3);

model = fit!(model, inputs, targets);

testOutputs = predict(model, test_inputs);

results = printConfusionMatrix(testOutputs, test_targets);
SVM_test_results = Dict(zip(keys_model,collect(results)));
#######################################################################################

#######################################################################################
# Modelo ANN 7
ann, = trainClassANN(topology7, (inputs, oneHotEncoding(targets)),
                    testDataset = (test_inputs, oneHotEncoding(test_targets));
                    maxEpochs=maxEpochs, learningRate=learningRate);

testOutputs = collect(ann(test_inputs')');

results = printConfusionMatrix(testOutputs, oneHotEncoding(test_targets));
ANN_test_results = Dict(zip(keys_model,collect(results)));
#######################################################################################

#######################################################################################
# Datos de precisión para cada modelo
kNN_results = [kNN_results1[1][1], kNN_results2[1][1], kNN_results3[1][1], kNN_results4[1][1], kNN_results5[1][1], kNN_results6[1][1]]
ANN_results = [ANN_results1[1][1], ANN_results2[1][1], ANN_results3[1][1], ANN_results4[1][1], ANN_results5[1][1], ANN_results6[1][1], ANN_results7[1][1], ANN_results8[1][1]]
SVM_results = [SVM_results1[1][1], SVM_results2[1][1], SVM_results3[1][1], SVM_results4[1][1], SVM_results5[1][1], SVM_results6[1][1], SVM_results7[1][1], SVM_results8[1][1]]
DecisionTree_results = [DecisionTree_results1[1][1], DecisionTree_results2[1][1], DecisionTree_results3[1][1], DecisionTree_results4[1][1], DecisionTree_results5[1][1], DecisionTree_results6[1][1]]
#######################################################################################

#######################################################################################
# Crear el diagrama de cajas
boxplot(["ANN" "Decision Tree" "SVM" "kNN"], [ANN_results, DecisionTree_results, SVM_results, kNN_results], xlabel="Modelos", ylabel="Precisión", title="Comparación de Modelos", legend=:none)
#######################################################################################