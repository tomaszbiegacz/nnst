local TransferFunctionReCU, parent = torch.class('nnst.TransferFunctionReCU', 'nnst.TransferFunction')

--[[
   Rectified Cubic Units (ReCU) transfer function
--]]

function TransferFunctionReCU:__init()
   parent.__init(self)
end

local function getReCU(x)
    if x >= 0 then
        return x*x*x
    else
        return 0
    end
end

local function gradReCU(x)
    if x >= 0 then
        return 3*x*x
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

function TransferFunctionReCU:updateOutput(input)
    self.output = input:clone()
    self.output:apply(getReCU)

    return self.output
end

function TransferFunctionReCU:updateGradInput(input, gradOutput)
    self.gradForInput = input:clone()
    self.gradForInput:apply(gradReCU)

    self.gradInput = torch.cmul(gradOutput, self.gradForInput)

    self.grad2ForInput = input:clone()
    self.grad2ForInput:apply(grad2ReCU)

    return self.gradInput
end

function TransferFunctionReCU:__tostring__()
  return string.format('%s', torch.type(self))
end
