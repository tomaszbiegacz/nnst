local TransferFunction, parent = torch.class('nnst.TransferFunction', 'nn.Module')

--[[
   NNST transfer function
--]]

function TransferFunction:__init()
   parent.__init(self)   

   self.gradForInput = torch.Tensor()
   self.grad2ForInput = torch.Tensor()

   self.grad2Input = torch.Tensor()
end

function TransferFunction:zeroGradParameters()
    parent.zeroGradParameters(self)

    self.gradInput:zero();
    self.gradForInput:zero();

    self.grad2Input:zero();
    self.grad2ForInput:zero();
end

function TransferFunction:updateGrad2Input(input, gradOutput, grad2Output)
    assert(input:dim() == 1)

    self.grad2Input:resizeAs(input)
    self.grad2Input:zero()

    local gradForInputSqr = torch.cmul(self.gradForInput, self.gradForInput)
    self.grad2Input:addcmul(grad2Output, gradForInputSqr)
    self.grad2Input:addcmul(gradOutput, self.grad2ForInput)
end

function TransferFunction:backward2(input, gradOutput, grad2Output)
    self:backward(input, gradOutput)

    self:updateGrad2Input(input, gradOutput, grad2Output)

    return self.gradInput, self.grad2Input
end

function TransferFunction:updateParametersNewton(learningRate)
    -- intentionally empty
end