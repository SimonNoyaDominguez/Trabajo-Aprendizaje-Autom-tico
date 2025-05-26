using FileIO
using Images
using Plots
using StatsBase
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
# Todas las imágenes en un mismo vector
imgsR = cat_vector_of_imgs(imgsRarr);
imgsL = cat_vector_of_imgs(imgsLarr);

imgs = vcat(imgsR, imgsL);

test_imgsR = cat_vector_of_imgs(test_imgsRarr);
test_imgsL = cat_vector_of_imgs(test_imgsLarr);

test_imgs = vcat(test_imgsR, test_imgsL);
#######################################################################################


#######################################################################################
# Se genera un vector donde cada elemento sera el número de dedos levantados
# correspondiente a cada instancia
targets_R = gen_targets(imgsRarr);
targets_L = gen_targets(imgsLarr);

test_targets_R = gen_targets(test_imgsRarr);
test_targets_L = gen_targets(test_imgsLarr);
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
# Tests para ver como funciona el tratamiento de imágenes
function show_bounding_box_and_print_ratio(num::Int, bool_imgs::Vector{BitMatrix}, imgs::Vector{Matrix{Gray{N0f8}}},
  labelsArrays::Vector{Matrix{Int64}}=Vector{Matrix{Int64}}();
  boundingBoxes::AbstractVector{<:AbstractVector{CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}}}=Vector{Vector{CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}}}(),
  filter_by_size::Int=0)
  if isempty(boundingBoxes)
    if isempty(labelsArrays)
      labelArray = ImageMorphology.label_components(bool_imgs[num]);
    else
      labelArray = labelsArrays[num]
    end
    if(filter_by_size != 0)
      sizes_labelArrayImgs = ImageMorphology.component_lengths(labelArray);
      labels_to_remove = get_labels_min_size_to_remove([sizes_labelArrayImgs], filter_by_size);
      labelArray = remove_labels_from_labelsArrays(labels_to_remove, [labelArray])[1];
    end

    boundingBoxes = ImageMorphology.component_boxes(labelArray);
    imagenObjetos1 = copy(imgs[num]);
    imagenObjetos2 = (Gray.(labelArray.==1));
    imagenesObjetos = [imagenObjetos1, imagenObjetos2];
  else
    boundingBoxes = boundingBoxes[num]
    imagenObjetos1 = copy(imgs[num]);
    imagenesObjetos = [imagenObjetos1];
  end
  for imagenObjetos in imagenesObjetos
    for i in eachindex(boundingBoxes)
      imagenObjetos[ boundingBoxes[i][:,1] ] .= Gray{N0f8}(1);
      imagenObjetos[ boundingBoxes[i][:,end] ] .= Gray{N0f8}(1);
      imagenObjetos[ boundingBoxes[i][1,:] ] .= Gray{N0f8}(1);
      imagenObjetos[ boundingBoxes[i][end,:] ] .= Gray{N0f8}(1);

      h = length(boundingBoxes[i][:,1]);
      w = length(boundingBoxes[i][1,:]);
      println("BoxNum= ", i)
      println("height= ", h)
      println("width= ", w)
      println("ratio= ", h/w)
      println()
    end;
    display(imagenObjetos);
  end
end
#######################################################################################

# topology = Int[3];
# validationDataset = (inputs[:,1:1], oneHotEncoding(targets));
# trainingDataset = (inputs[:,1:1], oneHotEncoding(targets));
# testDataset = (inputs[:,1:1], oneHotEncoding(targets));
# transferFunctions = fill(sigmoid, length(topology));
# maxEpochs = 500;
# minLoss = Float32(0.0);
# learningRate = Float32(0.01);
# maxEpochsVal = 20;
# showText = false;


# ann, losses, validationLosses, testLosses = trainClassANN(topology, trainingDataset; validationDataset, testDataset, transferFunctions, 
#                               maxEpochs, minLoss, learningRate, maxEpochsVal, showText);

# plot(1:length(losses),losses);
# plot!(validationLosses);
# plot!(testLosses);
# display(Plots.current())



# display(Gray.(bool_imgs[14000]))
# display(imgs[14000])
# display(targets[14000,:])

# num=234; show_bounding_box_and_print_ratio(num,bool_imgs, imgs); inputs[num]

# show_bounding_box_and_print_ratio(num,test_bool_imgs, test_imgs); test_inputs[num]



# Caso problemático
num=6000; show_bounding_box_and_print_ratio(num, bool_imgs, imgs);
num=6000; show_bounding_box_and_print_ratio(num, bool_imgs, imgs, labelsArraysImgs);
num=6000; show_bounding_box_and_print_ratio(num, bool_imgs, imgs, labelsArraysImgs; boundingBoxes=boundingBoxesImgs);
num=6000; show_bounding_box_and_print_ratio(num, bool_imgs, imgs; filter_by_size=1000);

get_count_horizontal_intersections(boxBoolImgs, 3/8)

# num=8000; ratio=3/8; l=Int(trunc(size(boxBoolImgs[num],1)*ratio)); maximum(ImageMorphology.label_components(boxBoolImgs[num][l:l,:]))
# Gray.(boxBoolImgs[num][l:l,:])