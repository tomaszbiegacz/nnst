local Linear, parent = torch.class('nnst.Linear', 'nn.Linear')

function Linear:__init(inputSize, outputSize, bias)
    parent.__init(self, inputSize, outputSize, bias)

    self.gradForInput = torch.Tensor()
    self.grad2ForInput = torch.Tensor()

    self.grad2Input = torch.Tensor()

    self.gradWeight:zero()
    self.grad2Weight = torch.Tensor():resizeAs(self.gradWeight)
    if self.bias then
        self.gradBias:zero()
        self.grad2Bias = torch.Tensor():resizeAs(self.gradBias)
    end
end

function Linear:updateGradInput(input, gradOutput)
    --
    -- for some reason "parent" is "nn.Module",
    -- hence I have to copy the code from "Linear" directly
    --

    local nElement = self.gradInput:nElement()
    self.gradInput:resizeAs(input)
    if self.gradInput:nElement() ~= nElement then
       self.gradInput:zero()
    end
    if input:dim() == 1 then
       self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
    elseif input:dim() == 2 then
       self.gradInput:addmm(0, 1, gradOutput, self.weight)
    end

    self.gradForInput = self.weight:clone()
    self.grad2ForInput = torch.Tensor():resizeAs(self.weight):zero()

    return self.gradInput
end

function Linear:zeroGradParameters()
    parent.zeroGradParameters(self)

    self.gradInput:zero();
    self.gradForInput:zero();

    self.grad2Input:zero();
    self.grad2ForInput:zero();

    self.grad2Weight:zero();
    if self.bias then
        self.grad2Bias:zero();
    end
end

--
-- NNST module
--

function Linear:updateGrad2Input(input, gradOutput, grad2Output)
    assert(input:dim() == 1)

    local nElement = self.grad2Input:nElement()
    self.grad2Input:resizeAs(input)
    if self.grad2Input:nElement() ~= nElement then
        self.grad2Input:zero()
    end

    local weightT = self.weight:t()
    local weightTsqr = torch.cmul(weightT, weightT)
    self.grad2Input:addmv(0, 1, weightTsqr, grad2Output)
end

function Linear:accGrad2Parameters(input, gradOutput, grad2Output, scale)
    assert(input:dim() == 1)

    scale = scale or 1

    local inputSqr = torch.cmul(input, input)
    self.grad2Weight:addr(scale, grad2Output, inputSqr)
    if self.bias then
        self.grad2Bias:add(scale, grad2Output)
    end
end

function Linear:backward2(input, gradOutput, grad2Output)
    self:backward(input, gradOutput)

    self:updateGrad2Input(input, gradOutput, grad2Output)
    self:accGrad2Parameters(input, gradOutput, grad2Output)

    return self.gradInput, self.grad2Input
end

function Linear:updateParametersNewton(learningRate)
    self.weight:add(-learningRate, torch.cdiv(self.gradWeight, torch.abs(self.grad2Weight)))
    if self.bias then
        self.bias:add(-learningRate, torch.cdiv(self.gradBias, torch.abs(self.grad2Bias)))
    end
end