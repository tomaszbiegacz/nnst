local ErrorCriterion, parent = torch.class('nnst.ErrorCriterion', 'nn.Criterion')

--[[
    Error criterion continous up to second derivative
--]]

function ErrorCriterion:__init()
    parent.__init(self)
end

function ErrorCriterion:updateGrad2Input(input, target)
    error("Abstract method")
end

function ErrorCriterion:backward2(input, target)
    local grad = self:backward(input, target)
    local grad2 = self:updateGrad2Input(input, target)

    return grad, grad2    
end