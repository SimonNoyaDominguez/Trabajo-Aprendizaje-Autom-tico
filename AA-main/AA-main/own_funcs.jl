
function readFilesInFolder(folder::String; extension::String="", ending::String="")
  fileNames = String[];
  for fileName in readdir(folder)
      cond_ext = isempty(extension) || ((length(fileName)>length(extension)+1) && (uppercase(fileName[end-length(extension):end]) == uppercase(string(".", extension))));
      # cond_end = isempty(ending) || ((length(fileName)>length(ending)) && (uppercase(fileName[end-length(ending)+1:end]) == uppercase(ending)));
      beforeDotPos = collect(findfirst(r".*\.", fileName))[end]-1;
      nameWithOutExt = fileName[1:beforeDotPos];
      cond_end = isempty(ending) || ((length(nameWithOutExt)>length(ending)) && (uppercase(nameWithOutExt[end-length(ending)+1:end]) == uppercase(ending)));
      if cond_ext && cond_end
          push!(fileNames, fileName);
      end;
  end;
  return fileNames;
end;

function loadImages(dir::String; extension::String="", ending::String="")
  path = [string(dir, value) for value in readFilesInFolder(dir; extension, ending)];
  imgs = [load(img) for img in path]

  return imgs;
end;

function imageToBoolM(img::Matrix{Gray{N0f8}}; threshold::Real=0.5)
  return (Float32.(img)).>threshold;

end;


function gen_targets(imgs::Vector{<:Vector{<:AbstractArray}}, indexes::AbstractVector{Int}=Int[])
  indexes = indexes.+1;
  if(!isempty(indexes))
    imgs = imgs[indexes]
  end
  num_pattern = length(imgs);
  imgs_per_pattern = length(imgs[1]);
  total_imgs = num_pattern * imgs_per_pattern;
  e = Array{Int64,1}(undef, total_imgs);
  for i in 0:num_pattern-1;
    e[i*imgs_per_pattern+1:(i+1)*imgs_per_pattern] .= i
  end
  return e;
end

function cat_vector_of_imgs(imgs::Vector{<:Vector{<:AbstractArray}}, indexes::AbstractVector{Int}=Int[])
  if(!isempty(indexes))
    indexes = indexes.+1;
    return reduce(vcat, imgs[indexes])
  else
    return reduce(vcat, imgs)
  end
end

function get_width_length_boxes_ratio(boxes::AbstractVector{<:AbstractVector{CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}}})
  function get_width_length_boxes_ratio_aux(box::AbstractVector{CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}})
    h = length(box[1][:,1]);
    w = length(box[1][1,:]);
    return Float32(h/w)
  end
  return get_width_length_boxes_ratio_aux.(boxes);
end

function get_box_img(m::AbstractMatrix, box::AbstractVector{CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}})
  return m[box[1]]
end

function get_ones_zeros_ratio(boxImgs::AbstractVector{<:AbstractMatrix})
  n_ones_zeros = countmap.(boxImgs);

  function get_ones_zeros_ratio_aux(d::Dict{Bool, Int64})
    return Float32(d[1]/d[0])
  end

  return get_ones_zeros_ratio_aux.(n_ones_zeros);
end

# Calcular que etiquetas son de objetos demasiado pequeños (100 pixeles cuadrados o menos, por defecto):
function get_labels_min_size_to_remove(sizes_labelArrays::AbstractVector{<:AbstractVector{Int64}}, minPx::Int=100 )
  function get_labels_min_size_to_remove_aux(size_labelArray::AbstractVector{Int64}, minPx::Int)
    return findall(size_labelArray .<= minPx)
  end
  return get_labels_min_size_to_remove_aux.(sizes_labelArrays, minPx)
end

function remove_labels_from_labelsArrays(labels_to_remove::Vector{Vector{Int64}}, labelArrays::Vector{Matrix{Int64}})
  function remove_labels_from_labelArray_aux(label_to_remove::Vector{Int64}, labelArray::Matrix{Int64})
    unique_values = unique(map(label -> (in(label,label_to_remove) && (label!=0)) ? 0 : label, unique(labelArray)));
    # Crear un diccionario para mapear los valores únicos a números secuenciales
    dict = Dict(value => i-1 for (i, value) in enumerate(unique_values))

    # Aplicar el diccionario para reemplazar los valores en la matriz y poner a 0 los que se eliminen
    return map(label -> (in(label,label_to_remove) && (label!=0)) ? 0 : dict[label] , labelArray)
  end

  return remove_labels_from_labelArray_aux.(labels_to_remove, labelArrays);
end

function get_count_horizontal_intersections(boxBoolImgs::Vector{BitMatrix}, ratio::Real=3/8)
  function get_count_horizontal_intersections_aux(boxBoolImg::BitMatrix, ratio::Real)
    ratio_aux = Int(trunc(size(boxBoolImg,1)*ratio));
    maximum(ImageMorphology.label_components(boxBoolImg[ratio_aux:ratio_aux,:]));
  end
  get_count_horizontal_intersections_aux.(boxBoolImgs, ratio)
end