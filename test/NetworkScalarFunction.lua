local json = require ("dkjson")
local tests = torch.TestSuite()

local precision = 1e-5

local function testJsonExportContent(n, expectedJson)
    local exportFilePath = os.tmpname()
    n:exportToJson(exportFilePath)

    fd = io.open(exportFilePath, 'r')
    local jsonExport = fd:read("*a")
    fd:close()

    os.remove (exportFilePath)

    local actualState, posActual, errActual = json.decode(jsonExport)
    local expectedState, posExpected, errExpected = json.decode(expectedJson)

    if errActual or errExpected then
      tester:assert(false, "JSON error")
    end

    tester:eq(actualState, expectedState)
end

local function importJson(jsonExport)
    local exportFilePath = os.tmpname()    
    fd = io.open(exportFilePath, 'w')
    fd:write(jsonExport)
    fd:close()

    local n = nnst.NetworkImporter:importFromJsonFile(exportFilePath)

    os.remove (exportFilePath)

    return n
end

function tests.ScalarFunctionNetwork_perceptonSchema()
    local n = nnst.NetworkScalarFunction()    
    
    tester:eq(n:getIsPercepton(), true)
    tester:eq(tostring(n), [[nn.Sequential {
  [input -> (1) -> (2) -> output]
  (1): nnst.Linear(1 -> 1)
  (2): nnst.TransferFunctionALU
}]])        
end

function tests.NetworkScalarFunction_perceptonSetup()
    local n = nnst.NetworkScalarFunction{transferFunction = "ReLU"}

    n:getInputLayer():setWeight(1, 1, -0.25417594378814)
    n:getInputLayer():setBias(1, 0.1411257237196)    

    testJsonExportContent(n, [=[{
  "formatName":"nnst.NetworkScalarFunction",
  "formatVersion":1,
  "perceptonsInLayerCount":1,
  "hiddenLayersCount":0,
  "transferFunction":"ReLU",
  "approximationNormalizer":"none",
  "layers":[{
      "bias":[0.1411257237196],
      "weight":[[-0.25417594378814]]
    }]
}]=])

    tester:eq(n:forwardScalars(torch.Tensor({1, -1})), torch.Tensor({0, 0.39530166750774}))
end


function tests.NetworkScalarFunction_perceptonImport()
    local n = importJson([=[{
  "formatName":"nnst.NetworkScalarFunction",
  "formatVersion":1,
  "perceptonsInLayerCount":1,
  "hiddenLayersCount":0,
  "transferFunction":"ReLU",
  "approximationNormalizer":"none",
  "layers":[{
      "bias":[0.1411257237196],
      "weight":[[-0.25417594378814]]
    }]
}]=])

    tester:eq(n.config.perceptonsInLayerCount, 1)
    tester:eq(n.config.hiddenLayersCount, 0)

    tester:eq(n:getInputLayer():getWeight(1, 1), -0.25417594378814)
    tester:eq(n:getInputLayer():getBias(1), 0.1411257237196)              

    n:zeroGradParameters()
    tester:eq(n:getStateDescription(), [[{ {
    bias = { 0.1411257237196 },
    grad2Bias = { 0 },
    grad2Input = {},
    grad2Weight = { { 0 } },
    gradBias = { 0 },
    gradInput = {},
    gradTransfer = {},
    gradWeight = { { 0 } },
    outLinear = {},
    outTransfer = {},
    weight = { { -0.25417594378814 } }
  } }]])    
  
    local values = n:forwardScalars(torch.Tensor({1, -1, -2}))    
    tester:eq(
      values, 
      torch.Tensor({0, 0.39530166750774, 0.64947761129588}))

    tester:eq(n:forwardScalar(-1), 0.39530166750774, precision)
    tester:eq(n:getOutputLayer():getLinearOutput(1), 0.39530166750774, precision)
    tester:eq(n:getOutputLayer():getTransferOutput(1), 0.39530166750774, precision)

    tester:eq(n:forwardScalar(1), 0, precision)
    tester:eq(n:getOutputLayer():getLinearOutput(1), -0.11305022006854, precision)
    tester:eq(n:getOutputLayer():getTransferOutput(1), 0, precision)
end

function tests.NetworkScalarFunction_perceptonImportWithNormalization()
    local n = importJson([=[{
  "formatName":"nnst.NetworkScalarFunction",
  "formatVersion":1,
  "perceptonsInLayerCount":1,
  "hiddenLayersCount":0,
  "transferFunction":"ReLU",
  "approximationNormalizer":"multiplicativeInverse",
  "layers":[{
      "bias":[0.1411257237196],
      "weight":[[-0.25417594378814]]
    }]
}]=])

    tester:eq(n.config.perceptonsInLayerCount, 1)
    tester:eq(n.config.hiddenLayersCount, 0)

    tester:eq(n:getInputLayer():getWeight(1, 1), -0.25417594378814)
    tester:eq(n:getInputLayer():getBias(1), 0.1411257237196)              

    n:zeroGradParameters()
    tester:eq(n:getState( { includeGrad = true } ), { {
    weight = { { -0.25417594378814 } },
    bias = { 0.1411257237196 },
    grad2Bias = { 0 },
    grad2Input = {},
    grad2Weight = { { 0 } },
    gradBias = { 0 },
    gradInput = {},
    gradTransfer = {},    
    gradWeight = { { 0 } }    
  } })
  
    local values = n:forwardScalars(torch.Tensor{-1.2, -2})    
    tester:eq(
      values, 
      torch.Tensor{0.35293901020972, 0.26821369561367},
      precision)
end

function tests.NetworkScalarFunction_simpleNetworkSchema()
    local n = nnst.NetworkScalarFunction{
        perceptonsInLayerCount = 2,
        hiddenLayersCount = 0
    }
    
    tester:eq(n:getIsPercepton(), false)
    tester:eq(tostring(n), [[nn.Sequential {
  [input -> (1) -> (2) -> (3) -> output]
  (1): nnst.Linear(1 -> 2)
  (2): nnst.TransferFunctionALU
  (3): nnst.Linear(2 -> 1) without bias
}]])        
end

function tests.NetworkScalarFunction_simpleNetworkImport()
    local n = importJson([=[{
  "formatName":"nnst.NetworkScalarFunction",
  "formatVersion":1,
  "perceptonsInLayerCount":2,
  "transferFunction":"TransferFunctionALU",
  "layers":[{
      "weight":[[-5],[-1.2]],
      "bias":[-7.7,-1.3]      
    },{
      "weight":[[-1,-1]]
    }]
}]=])

  tester:eq(n.config.perceptonsInLayerCount, 2)
  tester:eq(n.config.hiddenLayersCount, 0)

  tester:eq(n:getInputLayer():getBias(1), -7.7)
  tester:eq(n:getInputLayer():getWeight(1, 1), -5)
  
  tester:eq(n:getInputLayer():getBias(2), -1.3)
  tester:eq(n:getInputLayer():getWeight(2, 1), -1.2)  
    
  tester:eq(n:getOutputLayer():getWeight(1, 1), -1)
  tester:eq(n:getOutputLayer():getWeight(1, 2), -1)  
end


function tests.NetworkScalarFunction_networkSchema()
    local n = nnst.NetworkScalarFunction{
        perceptonsInLayerCount = 3,
        hiddenLayersCount = 1
    }
    
    tester:eq(n:getIsPercepton(), false)
    tester:eq(tostring(n), [[nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
  (1): nnst.Linear(1 -> 3)
  (2): nnst.TransferFunctionALU
  (3): nnst.Linear(3 -> 3)
  (4): nnst.TransferFunctionALU
  (5): nnst.Linear(3 -> 1) without bias
}]])
end

function tests.NetworkScalarFunction_networkSetup()
    local n = nnst.NetworkScalarFunction{
        perceptonsInLayerCount = 3,
        hiddenLayersCount = 1
    }

    n:getInputLayer():setBias(1, 0.53643726930022)
    n:getInputLayer():setWeight(1, 1, 0.2107952539809)
    
    n:getInputLayer():setBias(2, -0.75695079611614)
    n:getInputLayer():setWeight(2, 1, -0.012929839547724)
    
    n:getInputLayer():setBias(3, 0.087814401369542)
    n:getInputLayer():setWeight(3, 1, -0.85520229768008)
    
    n:getHiddenLayer(1):setBias(1, 0.49574572649041)
    n:getHiddenLayer(1):setWeight(1, 1, 0.19680656185845)
    n:getHiddenLayer(1):setWeight(1, 2, 0.31105222655341)
    n:getHiddenLayer(1):setWeight(1, 3, 0.012392923942345)

    n:getHiddenLayer(1):setBias(2, -0.27488900424205)
    n:getHiddenLayer(1):setWeight(2, 1, 0.47506363545032)
    n:getHiddenLayer(1):setWeight(2, 2, 0.47930901154687)
    n:getHiddenLayer(1):setWeight(2, 3, 0.066616545170467)

    n:getHiddenLayer(1):setBias(3, -0.49539361596011)
    n:getHiddenLayer(1):setWeight(3, 1, -0.45509714749044)
    n:getHiddenLayer(1):setWeight(3, 2, -0.20763624111525)
    n:getHiddenLayer(1):setWeight(3, 3, 0.041268623497876)

    n:getOutputLayer():setWeight(1, 1, 0.4064157584247)
    n:getOutputLayer():setWeight(1, 2, 0.15272706317715)
    n:getOutputLayer():setWeight(1, 3, -0.060302617401662)
        
    testJsonExportContent(n, [=[{
  "formatName":"nnst.NetworkScalarFunction",
  "formatVersion":1,
  "perceptonsInLayerCount":3,
  "hiddenLayersCount":1,
  "transferFunction":"TransferFunctionALU",      
  "approximationNormalizer":"none",
  "layers":[{
      "bias":[0.53643726930022,-0.75695079611614,0.087814401369542],
      "weight":[[0.2107952539809],[-0.012929839547724],[-0.85520229768008]]
    },{
      "bias":[0.49574572649041,-0.27488900424205,-0.49539361596011],
      "weight":[[0.19680656185845,0.31105222655341,0.012392923942345],[0.47506363545032,0.47930901154687,0.066616545170467],[-0.45509714749044,-0.20763624111525,0.041268623497876]]
    },{
      "weight":[[0.4064157584247,0.15272706317715,-0.060302617401662]]
    }]
}]=])
end

function tests.NetworkScalarFunction_networkImport()
    local n = importJson([=[{
  "formatName":"nnst.NetworkScalarFunction",
  "formatVersion":1,
  "perceptonsInLayerCount":3,
  "hiddenLayersCount":1,
  "transferFunction":"ReLU",      
  "layers":[{
      "bias":[0.53643726930022,-0.75695079611614,0.087814401369542],
      "weight":[[0.2107952539809],[-0.012929839547724],[-0.85520229768008]]
    },{
      "bias":[0.49574572649041,-0.27488900424205,-0.49539361596011],
      "weight":[[0.19680656185845,0.31105222655341,0.012392923942345],[0.47506363545032,0.47930901154687,0.066616545170467],[-0.45509714749044,-0.20763624111525,0.041268623497876]]
    },{
      "weight":[[0.4064157584247,0.15272706317715,-0.060302617401662]]
    }]
}]=])

    tester:eq(n.config.perceptonsInLayerCount, 3)
    tester:eq(n.config.hiddenLayersCount, 1)

    tester:eq(n:getInputLayer():getBias(1), 0.53643726930022)
    tester:eq(n:getInputLayer():getWeight(1, 1), 0.2107952539809)
    
    tester:eq(n:getInputLayer():getBias(2), -0.75695079611614)
    tester:eq(n:getInputLayer():getWeight(2, 1), -0.012929839547724)
    
    tester:eq(n:getInputLayer():getBias(3), 0.087814401369542)
    tester:eq(n:getInputLayer():getWeight(3, 1), -0.85520229768008)
    
    tester:eq(n:getHiddenLayer(1):getBias(1), 0.49574572649041)
    tester:eq(n:getHiddenLayer(1):getWeight(1, 1), 0.19680656185845)
    tester:eq(n:getHiddenLayer(1):getWeight(1, 2), 0.31105222655341)
    tester:eq(n:getHiddenLayer(1):getWeight(1, 3), 0.012392923942345)

    tester:eq(n:getHiddenLayer(1):getBias(2), -0.27488900424205)
    tester:eq(n:getHiddenLayer(1):getWeight(2, 1), 0.47506363545032)
    tester:eq(n:getHiddenLayer(1):getWeight(2, 2), 0.47930901154687)
    tester:eq(n:getHiddenLayer(1):getWeight(2, 3), 0.066616545170467)

    tester:eq(n:getHiddenLayer(1):getBias(3), -0.49539361596011)
    tester:eq(n:getHiddenLayer(1):getWeight(3, 1), -0.45509714749044)
    tester:eq(n:getHiddenLayer(1):getWeight(3, 2), -0.20763624111525)
    tester:eq(n:getHiddenLayer(1):getWeight(3, 3), 0.041268623497876)

    tester:eq(n:getOutputLayer():getWeight(1, 1), 0.4064157584247)
    tester:eq(n:getOutputLayer():getWeight(1, 2), 0.15272706317715)
    tester:eq(n:getOutputLayer():getWeight(1, 3), -0.060302617401662)

    n:zeroGradParameters()
    tester:eq(n:getState( { includeGrad = true } ), { {
    bias = { 0.53643726930022, -0.75695079611614, 0.087814401369542 },
    gradBias = { 0, 0, 0 },
    gradTransfer = {},
    gradInput = {},
    gradWeight = { { 0 }, { 0 }, { 0 } },
    grad2Input = {},
    grad2Bias = { 0, 0, 0 },
    grad2Weight = { { 0 }, { 0 }, { 0 } },
    weight = { { 0.2107952539809 }, { -0.012929839547724 }, { -0.85520229768008 } }
  }, {
    bias = { 0.49574572649041, -0.27488900424205, -0.49539361596011 },
    gradBias = { 0, 0, 0 },
    gradTransfer = {},
    gradInput = {},
    gradWeight = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
    grad2Input = {},
    grad2Bias = { 0, 0, 0 },
    grad2Weight = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
    weight = { { 0.19680656185845, 0.31105222655341, 0.012392923942345 }, { 0.47506363545032, 0.47930901154687, 0.066616545170467 }, { -0.45509714749044, -0.20763624111525, 0.041268623497876 } }
  }, {  
    gradWeight = { { 0, 0, 0 } },
    grad2Input = {},
    grad2Weight = { { 0, 0, 0 } },
    gradInput = {},
    weight = { { 0.4064157584247, 0.15272706317715, -0.060302617401662 } }
  } })    
end

return tests