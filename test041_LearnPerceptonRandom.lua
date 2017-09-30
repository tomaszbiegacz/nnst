require('init');

--[[
    Learning linear function via percepton 
    with training set having values in random order.
--]]

function learn(net, errCriterion, x, y, exporter)
  net:zeroGradParameters()

  local xInput = torch.Tensor{ x }
  local yActual = net:forward(xInput)
  local yExpected = torch.Tensor{ y }
  local error = errCriterion:forward(yActual, yExpected)  

  local de, dde = errCriterion:backward2(yActual, yExpected)
  net:backward2(xInput, de, dde)  

  exporter:addState(x, y, error, de[1], dde[1], net)
  net:updateParametersNewton()
end

function main()
  local net = nnst.NetworkImporter:importFromJson([=[{
    "formatName":"nnst.NetworkScalarFunction",
    "formatVersion":1,
    "layers":[{
        "weight":[[5]],
        "bias":[5]
      }]
  }]=])

  local errCriterion = nnst.ErrorCriterionSquareRelative()

  -- local maxSampleValue = 100  
  -- local x = torch.randperm(maxSampleValue)
  local x = torch.Tensor{20,86,8,92,46,14,35,82,33,16,52,36,63,93,66,10,83,32,27,77,65,1,2,71,81,90,48,69,60,91,62,79,44,88,30,45,74,94,21,22,39,5,95,49,96,13,41,58,38,89,64,26,37,100,97,43,47,67,72,29,23,78,19,4,75,7,59,24,34,25,85,56,99,84,98,87,50,6,17,18,55,68,76,51,31,42,80,3,53,11,54,61,28,12,40,70,57,15,73,9}
  local y = 2*x+1

  local exporter = nnst.ExportPercepton()

  local lessonsCount = 100
  for i=1,lessonsCount do
    learn(net, errCriterion, x[i], y[i], exporter)
  end

  exporter:exportToCsv('testResults/test041_LearnPerceptonRandom.csv')
end

main()
