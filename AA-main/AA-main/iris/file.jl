using DelimitedFiles
using Flux
include("../funcs.jl")

function loadDataDlm(path::String, delimiter::Char, numInputs::Integer)

  dataset = readdlm(path,delimiter);
  # dataset = readdlm("./iris/dataset/iris.data",',');

  inputs = Float32.(dataset[:,1:numInputs]);
  # inputs = Float32.(dataset[:,1:4]);

  targets = dataset[:,numInputs+1];
  # targets = dataset[:,5];

  @assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y
  salidas deseadas no tienen el mismo número de filas"

  print("dataset loaded\n")
  return (inputs, targets)
end

# Carga de datos del dataset
(inputs, targets) = loadDataDlm("./iris/dataset/iris.data", ',', 4);


# Normalizacion
inputs = normalizeMinMax(inputs);

# Codificacion targets
targets = oneHotEncoding(targets);


# Dividiendo muestras en train y test
inputs_train = inputs[isodd.(1:end),:];
inputs_test = inputs[iseven.(1:end),:];

targets_train = targets[isodd.(1:end),:];
targets_test = targets[iseven.(1:end),:];



topology = Int[3];
dataset = (inputs_train, targets_train);
transferFunctions = fill(sigmoid, length(topology));
maxEpochs = 500;
minLoss = Float32(0.0);
learningRate = Float32(0.01);

# Entrenamiento de la red con el optimizador

# trainClassANN(topology, dataset); es igual a
# trainClassANN(topology, dataset; fill(sigmoid, length(topology)), 1000, 0.0, 0.01);
ann, losses = trainClassANN(topology, dataset; transferFunctions, maxEpochs, minLoss, learningRate);


# resultados clasificacion de la ann
rst = ann(inputs_test');
# cls_rst = classifyOutputs(rst');

# Determinación de precisión de resultados
# accuracy(ann(inputs_train')', targets_train);
acc = accuracy(rst', targets_test);
display(acc)


# Plot losses
using Plots
plot(1:length(losses),losses);
display(Plots.current())

