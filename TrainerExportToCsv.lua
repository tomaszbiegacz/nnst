local TrainerExportToCsv = {}

local function getScalarFunctionCsvHeader(sampleExperiment)            
    local sampleNet = sampleExperiment.net

    local exportValues = ''
    local exportGradients = ''

    local prevLayerNeuronsCount = 1
    for layerPos=1,table.getn(sampleNet) do
        local layer = sampleNet[layerPos]
        local neuronsCount = table.getn(layer.bias)

        for neuronPos=1,neuronsCount do
            for sourcePos=1,prevLayerNeuronsCount do
                exportValues = exportValues .. "W" .. layerPos .. "n" .. neuronPos .. "s" .. sourcePos .. ","
                exportGradients = exportGradients .. "gW" .. layerPos .. "n" .. neuronPos .. "s" .. sourcePos .. ","
            end
            exportValues = exportValues .. "B" .. layerPos .. "n" .. neuronPos .. ","
            exportGradients = exportGradients .. "gB" .. layerPos .. "n" .. neuronPos .. ",".. "gT" .. layerPos .. "n" .. neuronPos .. ","
        end

        prevLayerNeuronsCount = neuronsCount
    end

    return "x,Yp,Ye,Ec," .. "," .. exportValues .. "," .. "learnRate,gEc," .. exportGradients
end

local function getScalarFunctionCsvLine(experiment)            
    local net = experiment.net

    local exportValues = ''
    local exportGradients = ''

    local prevLayerNeuronsCount = 1
    for layerPos=1,table.getn(net) do
        local layer = net[layerPos]
        local neuronsCount = table.getn(layer.bias)

        for neuronPos=1,neuronsCount do
            for sourcePos=1,prevLayerNeuronsCount do
                exportValues = exportValues .. layer.weight[neuronPos][sourcePos] .. ","
                exportGradients = exportGradients .. layer.gradWeight[neuronPos][sourcePos] .. ","
            end
            exportValues = exportValues .. layer.bias[neuronPos] .. ","
            exportGradients = exportGradients .. layer.gradBias[neuronPos] .. ",".. layer.gradTransfer[neuronPos] .. ","
        end

        prevLayerNeuronsCount = neuronsCount
    end

    return 
        experiment.input .. "," .. 
        experiment.valuePredicted .. "," .. 
        experiment.valueExpected .. "," ..        
        experiment.errorCriterion .. "," ..
        "," ..
        exportValues ..
        "," ..
        experiment.learningRate .. "," ..
        experiment.gradErrorCriterion[1] .. "," ..
        exportGradients
end

local function scalarFunctionWithSingleLayer(path, state)    
    fd = io.open(path, 'w')

    fd:write(getScalarFunctionCsvHeader(state[1]), "\n")
    for i=1,table.getn(state) do
        local experiment = state[i]
        if experiment.net then
            fd:write(getScalarFunctionCsvLine(experiment), "\n")
        end
    end

    fd:close()
end

function TrainerExportToCsv:scalarFunction(path, trainer)
    local states = trainer:getLearningHistory()

    assert(table.getn(states) > 1, "table.getn(state) > 1")

    local firstExperiment = states[1]
    local layers = firstExperiment.net;
    local layersCount = table.getn(layers)
    if layersCount == 1 then
        scalarFunctionWithSingleLayer(path, states)
    else
        error("Invalid number of layers: " .. layersCount)
    end

    print('Training details saved under ' .. path)
end

nnst.TrainerExportToCsv = TrainerExportToCsv