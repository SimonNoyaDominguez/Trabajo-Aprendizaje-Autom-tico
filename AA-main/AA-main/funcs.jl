using FileIO;
using DelimitedFiles;
using Statistics;
using Flux;
using Flux.Losses
using Random
using Random:seed!
using ScikitLearn
using ScikitLearn: fit!, predict
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

# ----------------------------------------------------------------------------------------------------------
# PRACTICA 2
# ----------------------------------------------------------------------------------------------------------

function oneHotEncoding(feature::AbstractArray{<:Any,1},
    classes::AbstractArray{<:Any,1})
    numClasses = length(classes);

    if numClasses==2
        # Si solo hay dos clases, se devuelve una matriz con una columna
        # feature = reshape(feature.==classes[1], :, 1);
        feature = feature .== classes[1]
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase
        # Cualquiera de estos dos tipos (Array{Bool,2} o BitArray{2}) vale perfectamente
        feature = feature .== reshape(classes, 1, :);
    end;
    return feature;
end;

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = unique(feature);
    return oneHotEncoding(feature, classes);
end;

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, :, 1);
end;

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    minValues = minimum(dataset, dims=1);
    maxValues = maximum(dataset, dims=1);
    return (minValues, maxValues);
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    avgValues = mean(dataset, dims=1);
    stdValues = std(dataset, dims=1);
    return (avgValues, stdValues);
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    dataset .-= minValues
    dataset ./= (maxValues .- minValues)
    dataset[:, vec(minValues.==maxValues)] .= 0
    return dataset
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, normalizationParameters)
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    newdataset = copy(dataset)
    return normalizeMinMax!(newdataset, normalizationParameters)
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    newdataset = copy(dataset)
    return normalizeMinMax!(newdataset)
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues, stdValues = normalizationParameters
    dataset .-= avgValues
    dataset ./= stdValues
    dataset[:, vec(stdValues.==0)] .= 0
    return dataset
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, normalizationParameters)
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    newdataset = copy(dataset)
    return normalizeZeroMean!(newdataset, normalizationParameters)
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    newdataset = copy(dataset)
    return normalizeZeroMean!(newdataset)
end

function classifyOutputs(outputs::AbstractArray{<:Real,1}, threshold::Real=0.5)
    return outputs .>=threshold
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}, threshold::Real=0.5)
    if size(outputs, 2) == 1
        # Si la matriz tiene una columna, convertirla en un vector y llamar a la función anterior
        vectorOutputs = outputs[:]
        binaryVector = classifyOutputs(vectorOutputs, threshold)
        return reshape(binaryVector, :, 1)
    else
        # Si la matriz tiene más de una columna, crear una matriz de valores booleanos del mismo tamaño
        # y, para cada fila, poner a true la columna con un valor mayor
        maxIndices = argmax(outputs, dims=2);
        binaryMatrix = falses(size(outputs));
        binaryMatrix[maxIndices] .= true;
        return binaryMatrix
    end
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert (size(outputs,1)==size(targets,1)) "Las matrices de salidas deseadas y
                            salidas emitidas no tienen el mismo número de filas"
    return mean(targets .== outputs)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert (size(outputs,1)==size(targets,1)) "Las matrices de salidas deseadas y
                            salidas emitidas no tienen el mismo número de filas"
    if size(targets, 2) == 1
        return accuracy(outputs[:,1], targets[:,1])
    else
        return mean(all(targets .== outputs, dims=2))
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}, threshold::Real=0.5)
    return accuracy(classifyOutputs(outputs, threshold), targets )
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}, threshold::Real=0.5)
    if size(targets, 2) == 1
        return accuracy(classifyOutputs(outputs, threshold), targets[:,1] )
    else
        return accuracy(classifyOutputs(outputs, threshold), targets )
    end
end

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(sigmoid, length(topology)))
    
    ann = Chain();
    numInputsLayer = numInputs;;
    numLayers = length(topology);
    for i in 1:numLayers
        ann = Chain(ann..., Dense(numInputsLayer, topology[i], transferFunctions[i]) );
        numInputsLayer = topology[i];
    end;
    numLastLayer = isempty(topology) ? numInputs : topology[end];
    if numOutputs == 1
        ann = Chain(ann..., Dense(numLastLayer, numOutputs, sigmoid));
    else
        ann = Chain(ann..., Dense(numLastLayer, numOutputs, identity), softmax);
    end;
    return ann;
end

loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(sigmoid, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    (inputs, targets) = dataset;

    numInputs = size(inputs, 2);
    numOutputs = size(targets, 2);

    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions);

    actualLoss = loss(ann, inputs', targets');
    losses = Float32[];

    push!(losses, actualLoss);

    opt_state = Flux.setup(Adam(learningRate), ann);
    
    while (maxEpochs > 0) && (minLoss < actualLoss)
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);
        actualLoss = loss(ann, inputs', targets');
        push!(losses, actualLoss);
        maxEpochs-=1;
    end
    return (ann, losses);

end

function trainClassANN(topology::AbstractArray{<:Int,1},
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunctions::AbstractArray{<:Function,1}=fill(sigmoid, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    dataset = (inputs, reshape(targets,:,1));
    
    return trainClassANN(topology, dataset; transferFunctions, maxEpochs, minLoss, learningRate);

end

# ----------------------------------------------------------------------------------------------------------
# PRACTICA 3
# ----------------------------------------------------------------------------------------------------------

function holdOut(N::Int, P::Real)
    if P < 0 || P > 1
        error("P debe estar entre 0 y 1")
    end
    
    indices = randperm(N)
    split_index = Int(round(N*(1-P)))
    
    train_indices = indices[1:split_index]
    test_indices = indices[(split_index+1):end]
    
    return train_indices, test_indices
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    if Pval < 0 || Pval > 1 || Ptest < 0 || Ptest > 1 || Pval + Ptest > 1
        error("Pval y Ptest deben estar entre 0 y 1 y su suma no debe ser mayor que 1")
    end
    
    (trainingValidationIndices, testIndices) = holdOut(N, Ptest)
    (trainingIndices, validationIndices) = holdOut(length(trainingValidationIndices), Pval*N/length(trainingValidationIndices))

    return (trainingValidationIndices[trainingIndices], trainingValidationIndices[validationIndices], testIndices)
end

function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(sigmoid, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, showText::Bool=false)
    
    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;
    
    if size(trainingInputs, 1) != size(trainingTargets, 1)
        error("El tamaño de trainingInputs y trainingTargets no coincide en la primera dimensión")
    end
    if size(testInputs, 1) != size(testTargets, 1)
        error("El tamaño de testInputs y testTargets no coincide en la primera dimensión")
    end
    if size(validationInputs, 1) != size(validationTargets, 1)
        error("El tamaño de validationInputs y validationTargets no coincide en la primera dimensión")
    end
    if !isempty(validationInputs) && size(trainingInputs, 2) != size(validationInputs, 2)
        error("El tamaño de trainingInputs y validationInputs no coincide en la segunda dimensión")
    end
    if !isempty(validationTargets) && size(trainingTargets, 2) != size(validationTargets, 2)
        error("El tamaño de trainingTargets y validationTargets no coincide en la segunda dimensión")
    end
    if !isempty(testInputs) && size(trainingInputs, 2) != size(testInputs, 2)
        error("El tamaño de trainingInputs y testInputs no coincide en la segunda dimensión")
    end
    if !isempty(testTargets) && size(trainingTargets, 2) != size(testTargets, 2)
        error("El tamaño de trainingTargets y testTargets no coincide en la segunda dimensión")
    end

    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions=transferFunctions);
    
    losses   = Float32[];
    validationLosses = Float32[];
    testLosses       = Float32[];

    numEpoch = 0;

    function calculateLossValues()
        actualLoss = loss(ann, trainingInputs', trainingTargets');
        showText && print("Epoch ", numEpoch, ": Training loss: ", actualLoss);
        push!(losses, actualLoss);
        if !isempty(validationInputs)
            validationLoss = loss(ann, validationInputs', validationTargets');
            showText && print(" - validation loss: ", validationLoss);
            push!(validationLosses, validationLoss);
        else
            validationLoss = NaN;
        end;
        if !isempty(testInputs)
            testLoss       = loss(ann, testInputs', testTargets');
            showText && print(" - test loss: ", testLoss);
            push!(testLosses, testLoss);
        else
            testLoss = NaN;
        end;
        showText && println("");
        return (actualLoss, validationLoss, testLoss);
    end;
    (actualLoss, validationLoss, _) = calculateLossValues();

    numEpochsValidation = 0; bestValidationLoss = validationLoss;
    bestANN = deepcopy(ann);

    opt_state = Flux.setup(Adam(learningRate), ann);
    
    while (numEpoch<maxEpochs) && (actualLoss>minLoss) && (numEpochsValidation<maxEpochsVal)
        
        Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt_state);

        numEpoch += 1;

        (actualLoss, validationLoss, _) = calculateLossValues();

        if (!isempty(validationInputs))
            if (validationLoss<bestValidationLoss)
                bestValidationLoss = validationLoss;
                numEpochsValidation = 0;
                bestANN = deepcopy(ann);
            else
                numEpochsValidation += 1;
            end;
        end;

    end;

    if isempty(validationInputs)
        bestANN = ann;
    end;
    return (bestANN, losses, validationLosses, testLosses);
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(sigmoid, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, showText::Bool=false)

    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;

    return trainClassANN(topology, (trainingInputs, reshape(trainingTargets, length(trainingTargets), 1)); validationDataset=(validationInputs, reshape(validationTargets, length(validationTargets), 1)), testDataset=(testInputs, reshape(testTargets, length(testTargets), 1)), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=showText);
end;

# # ----------------------------------------------------------------------------------------------------------
# # PRACTICA 4
# # ----------------------------------------------------------------------------------------------------------

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    
    VP = sum((outputs .== true) .& (targets .== true)) # Verdaderos positivos
    VN = sum((outputs .== false) .& (targets .== false)) # Verdaderos negativos
    FP = sum((outputs .== true) .& (targets .== false)) # Falsos positivos
    FN = sum((outputs .== false) .& (targets .== true)) # Falsos negativos

    confusion_matrix = [VN FP; FN VP]

    accuracy = (VP + VN) / (VP + VN + FN + FP)
    error_rate = 1. - accuracy
    
    if (VN==length(targets) || (VP==length(targets)))
        recall = 1.;
        valor_predictivo_positivo = 1.;
        specificity = 1.;
        valor_predictivo_negativo = 1.;
    else
        recall = (VP==VP==0.) ? 0. : VP / (VP + FN) # Tasa de verdaderos positivos
        valor_predictivo_positivo = (VP==FP==0.) ? 0. : VP / (VP + FP) # Precisión
        specificity = (VN==FP==0.) ? 0. : VN / (VN + FP) # Tasa de verdaderos negativos
        valor_predictivo_negativo = (VN==FN==0.) ? 0. : VN / (VN + FN) # Valor predictivo negativo
    end;

    F1 = recall==valor_predictivo_positivo==0. ? 0. : 2. * (valor_predictivo_positivo * recall) / (valor_predictivo_positivo + recall) # F1 score

    return (accuracy, error_rate, recall, specificity, valor_predictivo_positivo, valor_predictivo_negativo, F1, confusion_matrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5)

    return confusionMatrix(classifyOutputs(outputs, threshold), targets);
end
    
function printConfusionMatrix(outputs::AbstractArray{Bool,1},
    targets::AbstractArray{Bool,1})
    (acc, errorRate, recall, specificity, VPP, VPN, F1, confMatrix) = confusionMatrix(outputs, targets;);
    numClasses = size(confMatrix,1);
    @assert(numClasses==2);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    println(" - \t + \t|");
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Positive predictive value: ", VPP);
    println("Negative predictive value: ", VPN);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, VPP, VPN, F1, confMatrix);
end
function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    printConfusionMatrix(classifyOutputs(outputs, threshold), targets);
end



function confusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)

    @assert(size(outputs)==size(targets));
    numClasses = size(targets,2);
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    end;

    # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
    @assert(all(sum(outputs, dims=2).==1));
    # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
    recall      = zeros(numClasses);
    specificity = zeros(numClasses);
    VPP   = zeros(numClasses);
    VPN         = zeros(numClasses);
    F1          = zeros(numClasses);
    # Calculamos el numero de patrones de cada clase
    numInstancesFromEachClass = vec(sum(targets, dims=1));
    # Calculamos las metricas para cada clase, esto se haria con un bucle similar a "for numClass in 1:numClasses" que itere por todas las clases
    #  Sin embargo, solo hacemos este calculo para las clases que tengan algun patron
    #  Puede ocurrir que alguna clase no tenga patrones como consecuencia de haber dividido de forma aleatoria el conjunto de patrones entrenamiento/test
    #  En aquellas clases en las que no haya patrones, los valores de las metricas seran 0 (los vectores ya estan asignados), y no se tendran en cuenta a la hora de unir estas metricas

    for numClass in findall(numInstancesFromEachClass.>0)
        # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
        (_, _, recall[numClass], specificity[numClass], VPP[numClass], VPN[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
    end;
  
    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
    # Calculamos la matriz de confusión haciendo un bucle doble que itere sobre las clases
   
    for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
        # Igual que antes, ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
        confMatrix[numClassTarget, numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
    end;
   

    # Aplicamos las forma de combinar las metricas macro o weighted
    if weighted
        # Calculamos los valores de ponderacion para hacer el promedio
        weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
        recall      = sum(weights.*recall);
        specificity = sum(weights.*specificity);
        VPP   = sum(weights.*VPP);
        VPN         = sum(weights.*VPN);
        F1          = sum(weights.*F1);
    else
        # No realizo la media tal cual con la funcion mean, porque puede haber clases sin instancias
        #  En su lugar, realizo la media solamente de las clases que tengan instancias
        numClassesWithInstances = sum(numInstancesFromEachClass.>0);
        recall      = sum(recall)/numClassesWithInstances;
        specificity = sum(specificity)/numClassesWithInstances;
        VPP   = sum(VPP)/numClassesWithInstances;
        VPN         = sum(VPN)/numClassesWithInstances;
        F1          = sum(F1)/numClassesWithInstances;
    end;
    # Precision y tasa de error las calculamos con las funciones definidas previamente
    acc = accuracy(outputs, targets);
    errorRate = 1 - acc;

    return (acc, errorRate, recall, specificity, VPP, VPN, F1, confMatrix);
end

function confusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)

    return confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);
end
function confusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)

    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
    # @assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique([targets; outputs]);
    # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    if (length(classes) == 2) 
        return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes));
    else 
        return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
    end
    #return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end


function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    numClasses = size(targets, 2)
    if (numClasses==2)
        return printConfusionMatrix(outputs, targets);
    else 
        (acc, errorRate, recall, specificity, VPP, VPN, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
        writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
        writeHorizontalLine();
        print("\t| ");
        print.("Cl. ", 1:numClasses, "\t| ");
        println("");
        writeHorizontalLine();
        for numClassTarget in 1:numClasses
            # print.(confMatrix[numClassTarget,:], "\t");
            print("Cl. ", numClassTarget, "\t| ");
            print.(confMatrix[numClassTarget,:], "\t| ");
            println("");
            writeHorizontalLine();
        end;
        println("Accuracy: ", acc);
        println("Error rate: ", errorRate);
        println("Recall: ", recall);
        println("Specificity: ", specificity);
        println("Positive predictive value: ", VPP);
        println("Negative predictive value: ", VPN);
        println("F1-score: ", F1);
        return (acc, errorRate, recall, specificity, VPP, VPN, F1, confMatrix);
    end;   
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique([targets; outputs]);
    # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    if (length(classes) == 2) 
        return printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes));
    else 
        return printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
    end
end


# # ----------------------------------------------------------------------------------------------------------
# # PRACTICA 5
# # ----------------------------------------------------------------------------------------------------------


function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    numClasses = size(targets,2);
    indices = Array{Int64,1}(undef, size(targets,1));
    for numClass in 1:numClasses
        indices[targets[:,numClass]] = crossvalidation(sum(targets[:,numClass]), k);
    end;
    return indices;
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    classes = unique(targets);
    indices = Array{Int64,1}(undef, length(targets));
    for class in classes
        indicesThisClass = (targets .== class);
        indices[indicesThisClass] = crossvalidation(sum(indicesThisClass), k);
    end;
    return indices;
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    return crossvalidation(targets, k);
end

function ANNCrossValidation(topology::AbstractArray{<:Int,1}, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1}; numExecutions::Int=50, transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)

    targets = oneHotEncoding(targets);

    numFolds = maximum(crossValidationIndices);

    # Creamos los vectores para las metricas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testErrorRates = Array{Float64,1}(undef, numFolds);
    testRecalls = Array{Float64,1}(undef, numFolds);
    testSpecificities = Array{Float64,1}(undef, numFolds);
    testVPP = Array{Float64,1}(undef, numFolds);
    testVPN = Array{Float64,1}(undef, numFolds);
    testF1  = Array{Float64,1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Dividimos los datos en entrenamiento y test
        trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
        testInputs        = inputs[crossValidationIndices.==numFold,:];
        trainingTargets   = targets[crossValidationIndices.!=numFold,:];
        testTargets       = targets[crossValidationIndices.==numFold,:];

        # En el caso de entrenar una RNA, este proceso es no determinístico, por lo que es necesario repetirlo para cada fold
        # Para ello, se crean vectores adicionales para almacenar las metricas para cada entrenamiento
        testAccuraciesEachRepetition = Array{Float64,1}(undef, numExecutions);
        testErrorRatesEachRepetition = Array{Float64,1}(undef, numExecutions);
        testRecallsEachRepetition = Array{Float64,1}(undef, numExecutions);
        testSpecificitiesEachRepetition = Array{Float64,1}(undef, numExecutions);
        testVPPEeachRepetition = Array{Float64,1}(undef, numExecutions);
        testVPNEachRepetition = Array{Float64,1}(undef, numExecutions);
        testF1EachRepetition = Array{Float64,1}(undef, numExecutions);

        for numTraining in 1:numExecutions
            
            if validationRatio>0

                # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                #  Para ello, hacemos un hold out
                (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*size(trainingInputs,1)/size(inputs,1));
                # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

                # Entrenamos la RNA
                ann, = trainClassANN(topology, (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                    validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                    testDataset =       (testInputs,                          testTargets);
                    transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss = minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal);

            else

                # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test
                ann, = trainClassANN(topology, (trainingInputs, trainingTargets),
                    testDataset = (testInputs,     testTargets);
                    transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss = minLoss, learningRate=learningRate);

            end;

            # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
            (acc, errorRate, recall, specificity, VPP, VPN, F1, _) = confusionMatrix(ann(testInputs')', testTargets);

            # Almacenamos las metricas de este entrenamiento
            testAccuraciesEachRepetition[numTraining] = acc;
            testErrorRatesEachRepetition[numTraining] = errorRate;
            testRecallsEachRepetition[numTraining] = recall;
            testSpecificitiesEachRepetition[numTraining] = specificity;
            testVPPEeachRepetition[numTraining] = VPP;
            testVPNEachRepetition[numTraining] = VPN;
            testF1EachRepetition[numTraining] = F1;

        end;

        # Almacenamos las metricas
        testAccuracies[numFold] = mean(testAccuraciesEachRepetition);
        testErrorRates[numFold] = mean(testErrorRatesEachRepetition);
        testRecalls[numFold] = mean(testRecallsEachRepetition);
        testSpecificities[numFold] = mean(testSpecificitiesEachRepetition);
        testVPP[numFold] = mean(testVPPEeachRepetition);
        testVPN[numFold] = mean(testVPNEachRepetition);
        testF1[numFold] = mean(testF1EachRepetition);

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end;

    println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), " %, with a standard deviation of ", std(testAccuracies));
    println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), " %, with a standard deviation of ", std(testF1));

    return ((mean(testAccuracies), std(testAccuracies)), (mean(testErrorRates), std(testErrorRates)), (mean(testRecalls), std(testRecalls)),
            (mean(testSpecificities), std(testSpecificities)), (mean(testVPP), std(testVPP)), (mean(testVPN), std(testVPN)),
            (mean(testF1), std(testF1)));

end





# # ----------------------------------------------------------------------------------------------------------
# # PRACTICA 6
# # ----------------------------------------------------------------------------------------------------------


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1})
    
    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Que clases de salida tenemos
    # Es importante calcular esto primero porque se va a realizar codificacion one-hot-encoding varias veces, y el orden de las clases deberia ser el mismo siempre
    classes = unique(targets);

    # Primero codificamos las salidas deseadas en caso de entrenar RR.NN.AA.
    if modelType==:ANN
        targets = oneHotEncoding(targets, classes);
    end;

    numFolds = maximum(crossValidationIndices);

    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testErrorRates = Array{Float64,1}(undef, numFolds);
    testRecalls = Array{Float64,1}(undef, numFolds);
    testSpecificities = Array{Float64,1}(undef, numFolds);
    testVPP = Array{Float64,1}(undef, numFolds);
    testVPN = Array{Float64,1}(undef, numFolds);
    testF1  = Array{Float64,1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Si vamos a usar unos de estos 3 modelos
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold];
            testTargets       = targets[crossValidationIndices.==numFold];

            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["degree"], gamma=modelHyperparameters["gamma"], C=modelHyperparameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["max_depth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(n_neighbors=modelHyperparameters["n_neighbors"]);
            end;

            # Entrenamos el modelo con el conjunto de entrenamiento
            model = fit!(model, trainingInputs, trainingTargets);

            # Pasamos el conjunto de test
            testOutputs = predict(model, testInputs);
            
            # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
            (acc, errorRate, recall, specificity, VPP, VPN, F1, _) = confusionMatrix(testOutputs, testTargets);

        else

            # Vamos a usar RR.NN.AA.
            @assert(modelType==:ANN);

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold,:];
            testTargets       = targets[crossValidationIndices.==numFold,:];

            # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
            #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento
            testAccuraciesEachRepetition = Array{Float64,1}(undef, numExecutions);
            testErrorRatesEachRepetition = Array{Float64,1}(undef, numExecutions);
            testRecallsEachRepetition = Array{Float64,1}(undef, numExecutions);
            testSpecificitiesEachRepetition = Array{Float64,1}(undef, numExecutions);
            testVPPEeachRepetition = Array{Float64,1}(undef, numExecutions);
            testVPNEachRepetition = Array{Float64,1}(undef, numExecutions);
            testF1EachRepetition = Array{Float64,1}(undef, numExecutions);

            # Se entrena las veces que se haya indicado
            for numTraining in 1:modelHyperparameters["numExecutions"]

                if modelHyperparameters["validationRatio"]>0

                    # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                    #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                    #  Para ello, hacemos un hold out
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

                    # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"], (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                        validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                        testDataset =       (testInputs,                          testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

                else

                    # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                    #  teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"], (trainingInputs, trainingTargets),
                        testDataset = (testInputs,     testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"]);

                end;

                # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                (testAccuraciesEachRepetition[numTraining], testErrorRatesEachRepetition, testRecallsEachRepetition,
                 testSpecificitiesEachRepetition, testVPPEeachRepetition, testVPNEachRepetition,
                 testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            end;

            # Calculamos el valor promedio de todos los entrenamientos de este fold
            acc = mean(testAccuraciesEachRepetition);
            errorRate = mean(testErrorRatesEachRepetition);
            recall = mean(testRecallsEachRepetition);
            specificity = mean(testSpecificitiesEachRepetition);
            VPP = mean(testVPPEeachRepetition);
            VPN = mean(testVPNEachRepetition);
            F1 = mean(testF1EachRepetition);

        end;

        # Almacenamos las metricas
        testAccuracies[numFold] = acc;
        testErrorRates[numFold] = errorRate;
        testRecalls[numFold] = recall;
        testSpecificities[numFold] = specificity;
        testVPP[numFold] = VPP;
        testVPN[numFold] = VPN;
        testF1[numFold] = F1;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %");

    end; # for numFold in 1:numFolds

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), " %, with a standard deviation of ", std(testAccuracies));

    # return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));

    return ((mean(testAccuracies), std(testAccuracies)), (mean(testErrorRates), std(testErrorRates)), (mean(testRecalls), std(testRecalls)),
            (mean(testSpecificities), std(testSpecificities)), (mean(testVPP), std(testVPP)), (mean(testVPN), std(testVPN)),
            (mean(testF1), std(testF1)));
end

