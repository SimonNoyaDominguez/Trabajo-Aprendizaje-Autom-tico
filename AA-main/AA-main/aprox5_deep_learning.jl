using Images
using Random
using Random: seed!
seed!(1);
include("own_funcs.jl")
include("funcs.jl")
using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using JLD2, FileIO
using Statistics: mean
using Optimisers
using Optimisers: Adam



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
# s_height = 28;
# s_width = 28;

# train_imgs   = load("MNIST.jld2", "train_imgs");
# train_labels = load("MNIST.jld2", "train_labels");
# test_imgs    = load("MNIST.jld2", "test_imgs");
# test_labels  = load("MNIST.jld2", "test_labels");
# labels = 0:9; # Las etiquetas
#######################################################################################



#######################################################################################
s_height = 50;
s_width = 50;

resize_imgs = map(x->imresize(x,(s_height,s_width)), bool_imgs);
resize_test_imgs = map(x->imresize(x,(s_height,s_width)), test_bool_imgs);

train_imgs   = map(x -> Float32.(x), resize_imgs);
train_labels = targets;
test_imgs    = map(x -> Float32.(x), resize_test_imgs);
test_labels  = test_targets;
labels = 0:5; # Las etiquetas
#######################################################################################


# Tanto train_imgs como test_imgs son arrays de arrays bidimensionales (arrays de imagenes), es decir, son del tipo Array{Array{Float32,2},1}
#  Generalmente en Deep Learning los datos estan en tipo Float32 y no Float64, es decir, tienen menos precision
#  Esto se hace, entre otras cosas, porque las tarjetas gráficas (excepto las más recientes) suelen operar con este tipo de dato
#  Si se usa Float64 en lugar de Float32, el sistema irá mucho más lento porque tiene que hacer conversiones de Float64 a Float32

# Para procesar las imagenes con Deep Learning, hay que pasarlas una matriz en formato HWCN
#  Es decir, Height x Width x Channels x N
#  En el caso de esta base de datos
#   Height = s_height
#   Width = s_width
#   Channels = 1 -> son imagenes en escala de grises
#     Si fuesen en color, Channels = 3 (rojo, verde, azul)
# Esta conversion se puede hacer con la siguiente funcion:
function convertirArrayImagenesHWCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, s_height, s_width, 1, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(s_height,s_width)) "Las imagenes no tienen tamaño $(s_height)x$(s_width)";
        nuevoArray[:,:,1,i] .= imagenes[i][:,:];
    end;
    return nuevoArray;
end;
train_imgs = convertirArrayImagenesHWCN(train_imgs);
test_imgs = convertirArrayImagenesHWCN(test_imgs);

println("Tamaño de la matriz de entrenamiento: ", size(train_imgs))
println("Tamaño de la matriz de test:          ", size(test_imgs))


# Cuidado: en esta base de datos las imagenes ya estan con valores entre 0 y 1
# En otro caso, habria que normalizarlas
println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");



function train_cnn!(name_ann::String, ann, batch_size_arg::Integer)

    println("\n\n", "Inicializando entrenamiento de la ", name_ann)

    # Se establece la semilla para la generación de números pseudoaleatorios
    # para que cada vez que se ejecute el entrenamiento de una arquitectura
    # empiece de nuevo la generación y no tenga dependencia de haber generado
    # ya números aleatorios en anteriores entrenamientos, así cada entrenamiento
    # se puede ejecutar de forma independiente y generará los mismos resultados
    seed!(1);

    # Cuando se tienen tantos patrones de entrenamiento (en este caso 18000),
    #  generalmente no se entrena pasando todos los patrones y modificando el error
    #  En su lugar, el conjunto de entrenamiento se divide en subconjuntos (batches)
    #  y se van aplicando uno a uno

    # Hacemos los indices para las particiones
    # Cuantos patrones va a tener cada particion
    batch_size = batch_size_arg
    # Creamos los indices: partimos el vector 1:N en grupos de batch_size
    gruposIndicesBatch = Iterators.partition(1:size(train_imgs,4), batch_size);
    println("Se han creado ", length(gruposIndicesBatch), " grupos de indices para distribuir los patrones en batches de ", batch_size, " patrones");


    # Creamos el conjunto de entrenamiento: va a ser un vector de tuplas. Cada tupla va a tener
    #  Como primer elemento, las imagenes de ese batch
    #     train_imgs[:,:,:,indicesBatch]
    #  Como segundo elemento, las salidas deseadas (en booleano, codificadas con one-hot-encoding) de esas imagenes
    #     Para conseguir estas salidas deseadas, se hace una llamada a la funcion onehotbatch, que realiza un one-hot-encoding de las etiquetas que se le pasen como parametros
    #     onehotbatch(train_labels[indicesBatch], labels)
    #  Por tanto, cada batch será un par dado por
    #     (train_imgs[:,:,:,indicesBatch], onehotbatch(train_labels[indicesBatch], labels))
    # Sólo resta iterar por cada batch para construir el vector de batches
    train_set = [ (train_imgs[:,:,:,indicesBatch], onehotbatch(train_labels[indicesBatch], labels)) for indicesBatch in gruposIndicesBatch];

    # Creamos un batch similar, pero con todas las imagenes de test
    test_set = (test_imgs, onehotbatch(test_labels, labels));


    # # Hago esto simplemente para liberar memoria, las variables train_imgs y test_imgs ocupan mucho y ya no las vamos a usar
    # train_imgs = nothing;
    # test_imgs = nothing;
    # GC.gc(); # Pasar el recolector de basura


    # Vamos a probar la RNA capa por capa y poner algunos datos de cada capa
    # Usaremos como entrada varios patrones de un batch
    numBatchCoger = 1; numImagenEnEseBatch = [12, 6];
    # Para coger esos patrones de ese batch:
    #  train_set es un array de tuplas (una tupla por batch), donde, en cada tupla, el primer elemento son las entradas y el segundo las salidas deseadas
    #  Por tanto:
    #   train_set[numBatchCoger] -> La tupla del batch seleccionado
    #   train_set[numBatchCoger][1] -> El primer elemento de esa tupla, es decir, las entradas de ese batch
    #   train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch] -> Los patrones seleccionados de las entradas de ese batch
    entradaCapa = train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch];
    numCapas = length(Flux.params(ann));
    println("La RNA tiene ", numCapas, " capas:");
    for numCapa in 1:numCapas
        println("   Capa ", numCapa, ": ", ann[numCapa]);
        # Le pasamos la entrada a esta capa
        capa = ann[numCapa];
        salidaCapa = capa(entradaCapa);
        println("      La salida de esta capa tiene dimension ", size(salidaCapa));
        entradaCapa = salidaCapa;
    end

    # Sin embargo, para aplicar un patron no hace falta hacer todo eso.
    #  Se puede aplicar patrones a la RNA simplemente haciendo, por ejemplo
    # ann(train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch]);




    # Definimos la funcion de loss de forma similar a las prácticas de la asignatura
    loss(ann, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Para calcular la precisión, hacemos un "one cold encoding" de las salidas del modelo y de las salidas deseadas, y comparamos ambos vectores
    accuracy(batch) = mean(onecold(ann(batch[1])) .== onecold(batch[2]));
    # Un batch es una tupla (entradas, salidasDeseadas), asi que batch[1] son las entradas, y batch[2] son las salidas deseadas


    # Mostramos la precision antes de comenzar el entrenamiento:
    #  train_set es un array de batches
    #  accuracy recibe como parametro un batch
    #  accuracy.(train_set) hace un broadcast de la funcion accuracy a todos los elementos del array train_set
    #   y devuelve un array con los resultados
    #  Por tanto, mean(accuracy.(train_set)) calcula la precision promedia
    #   (no es totalmente preciso, porque el ultimo batch tiene menos elementos, pero es una diferencia baja)
    println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*mean(accuracy.(train_set)), " %");


    # Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
    eta = 0.001;
    opt_state = Flux.setup(Adam(eta), ann);


    println("Comenzando entrenamiento...")
    mejorPrecision = -Inf;
    criterioFin = false;
    numCiclo = 0;
    numCicloUltimaMejora = 0;
    mejorModelo = nothing;

    while !criterioFin

        # Se entrena un ciclo
        Flux.train!(loss, ann, train_set, opt_state);

        numCiclo += 1;

        # Se calcula la precision en el conjunto de entrenamiento:
        precisionEntrenamiento = mean(accuracy.(train_set));
        println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

        # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
        if (precisionEntrenamiento >= mejorPrecision)
            mejorPrecision = precisionEntrenamiento;
            precisionTest = accuracy(test_set);
            println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
            mejorModelo = deepcopy(ann);
            numCicloUltimaMejora = numCiclo;
        end

        # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
        if (numCiclo - numCicloUltimaMejora >= 5) && (eta > 1e-6)
            eta /= 10.0
            Optimisers.adjust!(opt_state, eta)
            println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", eta);
            numCicloUltimaMejora = numCiclo;
        end

        # Criterios de parada:

        # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
        if (precisionEntrenamiento >= 0.999)
            println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
            criterioFin = true;
        end

        # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
        if (numCiclo - numCicloUltimaMejora >= 10)
            println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
            criterioFin = true;
        end
    end
    precisionTest = accuracy(test_set);
    println("   Precision en el conjunto de test: ", 100*precisionTest, " %");
end


funcionTransferenciaCapasConvolucionales = relu;



# Definimos la red con la funcion Chain, que concatena distintas capas
ann1 = Chain(

    Conv((4, 4), 1=>32, pad=(2,2), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    Conv((4, 4), 32=>16, pad=(2,2), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),

    Dense((((s_width+1)÷2+1)÷2)*(((s_width+1)÷2+1)÷2)*16, length(labels)),

    softmax
)

train_cnn!("CNN-1", ann1, 1024)



# Definimos la red con la funcion Chain, que concatena distintas capas
ann2 = Chain(

    Conv((5, 5), 1=>16, pad=(2,2), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    Conv((3, 3), 16=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),

    Dense((s_height÷4)*(s_width÷4)*16, length(labels)),

    softmax
)

train_cnn!("CNN-2", ann2, 1024)



# Definimos la red con la funcion Chain, que concatena distintas capas
ann3 = Chain(

    Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    Conv((5, 5), 16=>16, pad=(2,2), funcionTransferenciaCapasConvolucionales),

    MaxPool((4,4)),

    x -> reshape(x, :, size(x, 4)),

    Dense((s_height÷8)*(s_width÷8)*16, length(labels)),

    softmax
)

train_cnn!("CNN-3", ann3, 512)



# Definimos la red con la funcion Chain, que concatena distintas capas
ann4 = Chain(

    Conv((5, 5), 1=>32, pad=(2,2), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    Conv((3, 3), 32=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((4,4)),

    x -> reshape(x, :, size(x, 4)),

    Dense((s_height÷8)*(s_width÷8)*16, length(labels)),

    softmax
)

train_cnn!("CNN-4", ann4, 512)



# Definimos la red con la funcion Chain, que concatena distintas capas
ann5 = Chain(

    Conv((3, 3), 1=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    Conv((3, 3), 32=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((4,4)),

    x -> reshape(x, :, size(x, 4)),

    Dense((s_height÷8)*(s_width÷8)*16, length(labels)),

    softmax
)

train_cnn!("CNN-5", ann5, 512)



# Definimos la red con la funcion Chain, que concatena distintas capas
ann6 = Chain(

    Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((4,4)),

    x -> reshape(x, :, size(x, 4)),

    Dense((s_height÷8)*(s_width÷8)*32, length(labels)),

    softmax
)

train_cnn!("CNN-6", ann6, 512)



# Definimos la red con la funcion Chain, que concatena distintas capas
ann7 = Chain(

    Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    Conv((3, 3), 16=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((4,4)),

    x -> reshape(x, :, size(x, 4)),

    Dense((s_height÷8)*(s_width÷8)*16, length(labels)),

    softmax
)

train_cnn!("CNN-7", ann7, 512)



# Definimos la red con la funcion Chain, que concatena distintas capas
ann8 = Chain(

    Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    Conv((3, 3), 16=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),

    Dense((s_height÷4)*(s_width÷4)*16, length(labels)),

    softmax
)

train_cnn!("CNN-8", ann8, 512)



# Definimos la red con la funcion Chain, que concatena distintas capas
ann9 = Chain(

    Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    Conv((3, 3), 16=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    Conv((3, 3), 16=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),

    Dense((s_height÷8)*(s_width÷8)*16, length(labels)),

    softmax
)

train_cnn!("CNN-9", ann9, 512)



# Definimos la red con la funcion Chain, que concatena distintas capas
ann10 = Chain(

    # Primera capa: convolucion, que opera sobre una imagen s_height x s_width
    # Argumentos:
    #  (3, 3): Tamaño del filtro de convolucion
    #  1=>16:
    #   1 canal de entrada: una imagen (matriz) de entradas
    #      En este caso, hay un canal de entrada porque es una imagen en escala de grises
    #      Si fuese, por ejemplo, una imagen en RGB, serian 3 canales de entrada
    #   16 canales de salida: se generan 16 filtros
    #  Es decir, se generan 16 imagenes a partir de la imagen original con filtros 3x3
    # Entradas a esta capa: matriz 4D de dimension s_height x s_width x 1canal    x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension s_height x s_width x 16canales x <numPatrones>
    Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    # Capa maxpool: es una funcion
    # Divide el tamaño en 2 en el eje x y en el eje y: Pasa imagenes (s_height x s_width) a (s_height/2 x s_width/2)
    # Entradas a esta capa: matriz 4D de dimension s_height x s_width x 16canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension (s_height/2) x (s_width/2) x 16canales x <numPatrones>
    MaxPool((2,2)),

    # Tercera capa: segunda convolucion: Le llegan 16 imagenes de tamaño (s_height/2) x (s_width/2)
    #  16=>32:
    #   16 canales de entrada: 16 imagenes (matrices) de entradas
    #   32 canales de salida: se generan 32 filtros (cada uno toma entradas de 16 imagenes)
    #  Es decir, se generan 32 imagenes a partir de las 16 imagenes de entrada con filtros 3x3
    # Entradas a esta capa: matriz 4D de dimension (s_height/2) x (s_width/2) x 16canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension (s_height/2) x (s_width/2) x 32canales x <numPatrones>
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    # Capa maxpool: es una funcion
    # Divide el tamaño en 2 en el eje x y en el eje y: Pasa imagenes (s_height/2 x s_width/2) a (s_height/2^2 x s_width/2^2)
    # Entradas a esta capa: matriz 4D de dimension (s_height/2) x (s_width/2) x 32canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension  (s_height/2^2) x (s_width/2^2) x 32canales x <numPatrones>
    MaxPool((2,2)),

    # Tercera convolucion, le llegan 32 imagenes de tamaño (s_height/2^2 x s_width/2^2)
    #  32=>32:
    #   32 canales de entrada: 32 imagenes (matrices) de entradas
    #   32 canales de salida: se generan 32 filtros (cada uno toma entradas de 32 imagenes)
    #  Es decir, se generan 32 imagenes a partir de las 32 imagenes de entrada con filtros 3x3
    # Entradas a esta capa: matriz 4D de dimension (s_height/2^2) x (s_width/2^2) x 32canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension (s_height/2^2) x (s_width/2^2) x 32canales x <numPatrones>
    Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    # Capa maxpool: es una funcion
    # Divide el tamaño en 2 en el eje x y en el eje y: Pasa imagenes (s_height/2^2 x s_width/2^2) a (s_height/2^3 x s_width/2^3)
    # Entradas a esta capa: matriz 4D de dimension (s_height/2^2) x (s_width/2^2) x 32canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension (s_height/2^3) x (s_width/2^3) x 32canales x <numPatrones>
    MaxPool((2,2)),

    # Cambia el tamaño del tensot 3D en uno 2D
    #  Pasa matrices H x W x C x N a matrices H*W*C x N
    #  Es decir, cada patron de tamaño (s_height/2^3 x s_width/2^3) x 32 lo convierte en un array de longitud 3*3*32
    # Entradas a esta capa: matriz 4D de dimension (s_height/2^3) x (s_width/2^3) x 32canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension (s_height/2^3 x s_width/2^3 x 32) x <numPatrones>
    x -> reshape(x, :, size(x, 4)),

    # Capa totalmente conectada
    #  Como una capa oculta de un perceptron multicapa "clasico"
    #  Parametros: numero de entradas (s_height/2^3 x s_width/2^3 x 32) y numero de salidas (6)
    #   Se toman 10 salidas porque tenemos 6 clases (numeros de 0 a 5)
    # Entradas a esta capa: matriz 4D de dimension (s_height/2^3 x s_width/2^3 x 32) x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension  6 x <numPatrones>
    Dense((s_height÷8)*(s_width÷8)*32, length(labels)),

    # Finalmente, capa softmax
    #  Toma las salidas de la capa anterior y aplica la funcion softmax de tal manera
    #   que las 10 salidas sean valores entre 0 y 1 con las probabilidades de que un patron
    #   sea de una clase determinada (es decir, las probabilidades de que sea un digito determinado)
    #  Y, ademas, la suma de estas probabilidades sea igual a 1
    softmax

    # Cuidado: En esta RNA se aplica la funcion softmax al final porque se tienen varias clases
    # Si sólo se tuviesen 2 clases, solo se tiene una salida, y no seria necesario utilizar la funcion softmax
    #  En su lugar, la capa totalmente conectada tendria como funcion de transferencia una sigmoidal (devuelve valores entre 0 y 1)
    #  Es decir, no habria capa softmax, y la capa totalmente conectada seria la ultima, y seria Dense((s_height/2^3 x s_width/2^3 x 32), 1, σ)

)

train_cnn!("CNN-10", ann10, 512)


# Para volver a calcular precisión fuera de la función
accuracy(ann, imgs, labels) = mean(onecold(ann(imgs)) .== (labels.+1))
accuracy(ann4, train_imgs, train_labels)*100


# Se calcula la matriz de confusion del mejor modelo
keys_model = ["acc", "errorRate", "recall", "specificity", "VPP", "VPN", "F1", "confMatrix"];

testOutputs = collect(ann10(test_imgs)');

results = printConfusionMatrix(testOutputs, oneHotEncoding(test_targets));
ANN_test_results = Dict(zip(keys_model,collect(results)));
# ANN_test_results["confMatrix"]