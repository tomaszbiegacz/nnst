local TransferFunctionALU, parent = torch.class('nnst.TransferFunctionALU', 'nnst.TransferFunction')

--[[
   Arctangens Linear Units (ALU) transfer function
--]]

function TransferFunctionALU:__init()
   parent.__init(self)
end

local function getALU(x)
    if x >= 0 then
        return x
    else
        return math.atan(x)
    end
end

local function gradALU(x)
    if x >= 0 then
        return 1
    else
        local denominator = 1 + x*x
        return 1 / denominator
    end
end

local function grad2ALU(x)
    if x >= 0 then
        return 0
    else
        local denominator = 1 + x*x
        return -1*2*x/denominator/denominator
    end
end

function TransferFunctionALU:updateOutput(input)
    self.output = input:clone()
    self.output:apply(getALU)

    return self.output
end

function TransferFunctionALU:updateGradInput(input, gradOutput)
    self.gradForInput = input:clone()
    self.gradForInput:apply(gradALU)

    self.gradInput = torch.cmul(gradOutput, self.gradForInput)

    self.grad2ForInput = input:clone()
    self.grad2ForInput:apply(grad2ALU)

    return self.gradInput
end

function TransferFunctionALU:__tostring__()
  return string.format('%s', torch.type(self))
end
