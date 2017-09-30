local tests = torch.TestSuite()

local precision = 1e-10
local precision2ndOrder = 1e-2

local function getSquare(x)
  return x*x
end

local function getCubic(x)
  return x*x*x
end

local function getReCU(x)
    if x >= 0 then
        return getCubic(x)
    else
        return 0
    end
end

local function gradReCU(x)
    if x >= 0 then
        return 3*getSquare(x)
    else
        return 0
    end
end

local function grad2ReCU(x)
    if x >= 0 then
        return 6*x
    else
        return 0
    end
end

local function createScalarNetwork_1hiddenLayer(inputCount, hiddenPerceptonsCount)
  local n = nnst.NetworkSequential()
  n:_addPerceptonLayer(
    nnst.Linear(inputCount, hiddenPerceptonsCount), 
    nnst.TransferFunctionReCU())
  n:_addPerceptonLayer(
    nnst.Linear(hiddenPerceptonsCount, hiddenPerceptonsCount), 
    nnst.TransferFunctionReCU())
  n:_addPerceptonLayer(nnst.Linear(hiddenPerceptonsCount, 1, false))

  local e = nnst.ErrorCriterionSquare()
  local y0 = torch.rand(2)
  local y_expected = torch.rand(1) + 1

  local y_actual = n:forward(y0)
  local e_actual = e:forward(y_actual, y_expected)

  local de_actual, dde_actual = e:backward2(y_actual, y_expected)
  n:backward2(y0, de_actual, dde_actual)

  return n, y0, y_expected, y_actual, e_actual
end

function tests.NetworkSequential_scallarTest_2in_1out_3in1hiddenLayer()  

  local n, y0, y_expected, y_actual, e_actual = createScalarNetwork_1hiddenLayer(2, 3)
  while math.abs(n:getInputLayer():getWeightGrad(1, 1)) < precision and math.abs(n:getInputLayer():getBiasGrad(1, 1)) < precision do
    n, y0, y_expected, y_actual, e_actual = createScalarNetwork_1hiddenLayer(2, 3)
  end

  local nl1 = n:getInputLayer()
  local nl2 = n:getHiddenLayer(1)
  local nl3 = n:getOutputLayer()

  local de_dw111_actual = nl1:getWeightGrad(1, 1)
  local de_db11_actual = nl1:getBiasGrad(1)

  local dde_ddw111_actual = nl1:getWeightGrad2(1, 1)
  local dde_ddb11_actual = nl1:getBiasGrad2(1)

  --
  -- Output
  --  

  local s1 = torch.Tensor{
    nl1:getWeight(1, 1)*y0[1] + nl1:getWeight(1, 2)*y0[2] + nl1:getBias(1),
    nl1:getWeight(2, 1)*y0[1] + nl1:getWeight(2, 2)*y0[2] + nl1:getBias(2),
    nl1:getWeight(3, 1)*y0[1] + nl1:getWeight(3, 2)*y0[2] + nl1:getBias(3)
  }
  local y1 = torch.Tensor{ getReCU(s1[1]), getReCU(s1[2]), getReCU(s1[3]) }

  local s2 = torch.Tensor{
    nl2:getWeight(1, 1)*y1[1] + nl2:getWeight(1, 2)*y1[2] + nl2:getWeight(1, 3)*y1[3] + nl2:getBias(1),
    nl2:getWeight(2, 1)*y1[1] + nl2:getWeight(2, 2)*y1[2] + nl2:getWeight(2, 3)*y1[3] + nl2:getBias(2),
    nl2:getWeight(3, 1)*y1[1] + nl2:getWeight(3, 2)*y1[2] + nl2:getWeight(3, 3)*y1[3] + nl2:getBias(3)
  }
  local y2 = torch.Tensor{ getReCU(s2[1]), getReCU(s2[2]), getReCU(s2[3]) }

  local y3 = torch.Tensor{
    nl3:getWeight(1, 1)*y2[1] + nl3:getWeight(1, 2)*y2[2] + nl3:getWeight(1, 3)*y2[3]
  }
  local e_expected = getSquare(y3[1] - y_expected[1])
  
  tester:eq(y3, y_actual, precision)
  tester:eq(e_expected, e_actual, precision)  

  --
  -- 1st weight derrivative
  --  

  local de_expected = 2*(y3[1] - y_expected[1])

  local dy1_dw111 = torch.Tensor{ gradReCU(s1[1])*(y0[1] + 0 + 0), 0, 0 }
  local dy2_dw111 = torch.Tensor{
    gradReCU(s2[1])*(nl2:getWeight(1, 1)*dy1_dw111[1] + 0 + 0),
    gradReCU(s2[2])*(nl2:getWeight(2, 1)*dy1_dw111[1] + 0 + 0),
    gradReCU(s2[3])*(nl2:getWeight(3, 1)*dy1_dw111[1] + 0 + 0),
  }
  local dy3_dw111 = torch.Tensor{
    nl3:getWeight(1, 1)*dy2_dw111[1] + nl3:getWeight(1, 2)*dy2_dw111[2] + nl3:getWeight(1, 3)*dy2_dw111[3]
  }
  local de_dw111_expected = de_expected*dy3_dw111[1]    
  
  tester:eq(de_dw111_expected, de_dw111_actual, precision)

  --
  -- 1st bias derrivative
  --

  local dy1_db11 = torch.Tensor{ gradReCU(s1[1]), 0, 0 }
  local dy2_db11 = torch.Tensor{
    gradReCU(s2[1])*(nl2:getWeight(1, 1)*dy1_db11[1] + 0 + 0),
    gradReCU(s2[2])*(nl2:getWeight(2, 1)*dy1_db11[1] + 0 + 0),
    gradReCU(s2[3])*(nl2:getWeight(3, 1)*dy1_db11[1] + 0 + 0),
  }
  local dy3_db11 = torch.Tensor{
    nl3:getWeight(1, 1)*dy2_db11[1] + nl3:getWeight(1, 2)*dy2_db11[2] + nl3:getWeight(1, 3)*dy2_db11[3]
  }
  local de_db11_expected = de_expected*dy3_db11[1]    

  tester:eq(de_db11_expected, de_db11_actual, precision)  

  --
  -- 2nd weight derrivative
  --  

  local dde_expected = 2

  local ddy1_ddw111 = torch.Tensor{ grad2ReCU(s1[1])*getSquare(y0[1]), 0, 0 }
  local ddy2_ddw111 = torch.Tensor{
    grad2ReCU(s2[1])*getSquare(nl2:getWeight(1, 1)*dy1_dw111[1]) + gradReCU(s2[1])*nl2:getWeight(1, 1)*ddy1_ddw111[1],
    grad2ReCU(s2[2])*getSquare(nl2:getWeight(2, 1)*dy1_dw111[1]) + gradReCU(s2[2])*nl2:getWeight(2, 1)*ddy1_ddw111[1],
    grad2ReCU(s2[3])*getSquare(nl2:getWeight(3, 1)*dy1_dw111[1]) + gradReCU(s2[3])*nl2:getWeight(3, 1)*ddy1_ddw111[1],
  }
  local ddy3_ddw111 = torch.Tensor{
    nl3:getWeight(1, 1)*ddy2_ddw111[1] + nl3:getWeight(1, 2)*ddy2_ddw111[2] + nl3:getWeight(1, 3)*ddy2_ddw111[3]
  }
  local dde_ddw111_expected = dde_expected*getSquare(dy3_dw111[1]) + de_expected*ddy3_ddw111[1]
  
  local ddw111_relativeError = (dde_ddw111_expected - dde_ddw111_actual) / math.min(dde_ddw111_expected, dde_ddw111_actual)
  if math.abs(ddw111_relativeError) > precision2ndOrder then
    print("ddw111_relativeError: " .. ddw111_relativeError)
    tester:eq(dde_ddw111_expected, dde_ddw111_actual, precision)
  end

  --
  -- 2nd bias derrivative
  --    

  local ddy1_ddb11 = torch.Tensor{ grad2ReCU(s1[1]), 0, 0 }
  local ddy2_ddb11 = torch.Tensor{
    grad2ReCU(s2[1])*getSquare(nl2:getWeight(1, 1)*dy1_db11[1]) + gradReCU(s2[1])*nl2:getWeight(1, 1)*ddy1_ddb11[1],
    grad2ReCU(s2[2])*getSquare(nl2:getWeight(2, 1)*dy1_db11[1]) + gradReCU(s2[2])*nl2:getWeight(2, 1)*ddy1_ddb11[1],
    grad2ReCU(s2[3])*getSquare(nl2:getWeight(2, 1)*dy1_db11[1]) + gradReCU(s2[3])*nl2:getWeight(3, 1)*ddy1_ddb11[1]
  }
  local ddy3_ddb11 = torch.Tensor{
    nl3:getWeight(1, 1)*ddy2_ddb11[1] + nl3:getWeight(1, 2)*ddy2_ddb11[2] + nl3:getWeight(1, 3)*ddy2_ddb11[3]
  }
  local dde_ddb11_expected = dde_expected*getSquare(dy3_db11[1]) + de_expected*ddy3_ddb11[1]
  
  local ddb11_relativeError = (dde_ddb11_expected - dde_ddb11_actual) / math.min(dde_ddb11_expected, dde_ddb11_actual)
  if math.abs(ddb11_relativeError) > precision2ndOrder then
    print("ddb11_relativeError: " .. ddb11_relativeError)
    tester:eq(dde_ddb11_expected, dde_ddb11_actual, precision)
  end

end

return tests
