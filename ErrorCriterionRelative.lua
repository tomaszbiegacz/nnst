local ErrorCriterionRelative, parent = torch.class('nnst.ErrorCriterionRelative', 'nnst.ErrorCriterion')

--[[
    Relative error criterion continous up to second derivative
--]]

function ErrorCriterionRelative:__init(paramAlpha, paramBeta)
    parent.__init(self)

    self.alpha = paramAlpha or 1
    self.beta = paramBeta or self.alpha/5

    assert(self.alpha > 0)
    assert(self.beta > 0)
    assert(self.alpha > self.beta)

    self.alpha_m_beta = self.alpha - self.beta
    self.alpha_p_beta = self.alpha + self.beta
    self.alpha2_m_beta2 = self.alpha_m_beta*self.alpha_p_beta

    self.q = self.beta + (self.alpha_m_beta*self.alpha_m_beta-3*self.beta*self.beta) / (3*self.alpha_p_beta)
end

local function isLayoutReversed(e)
    return e < 0
end

local function getLinearFuntionBounduaries(params, e)
    if isLayoutReversed(e) then
        return {
            R = e * (1 - params.alpha),
            L = e * (1 + params.alpha)
        }
    else
        return {
            L = e * (1 - params.alpha),
            R = e * (1 + params.alpha)
        }
    end
end

local function addCubicFunction(result)
    function result:eval(x)
        return self.a*x*x*x + self.b*x*x + self.c*x + self.d
    end

    function result:evalGrad(x)
        return 3*self.a*x*x + 2*self.b*x + self.c
    end

    function result:evalGrad2(x)
        return 6*self.a*x + 2*self.b
    end

    return result
end

local function getPositiveLeftCubicFunction(params, e)
    local e2 = e*e
    local e3 = e2*e
    local v1_m_alpha = 1 - params.alpha

    local result = {
        a = 1 / (3*e3*params.alpha2_m_beta2),
        b = -1 * v1_m_alpha / (e2*params.alpha2_m_beta2),
        c = -1/e + v1_m_alpha*v1_m_alpha / (e*params.alpha2_m_beta2),
        d = 1 - v1_m_alpha*v1_m_alpha*v1_m_alpha / (3*params.alpha2_m_beta2)
    }

    return addCubicFunction(result)
end

local function getPositiveRightCubicFunction(params, e)
    local e2 = e*e
    local e3 = e2*e
    local v1_p_alpha = 1 + params.alpha

    local result = {
        a = -1 / (3*e3*params.alpha2_m_beta2),
        b = v1_p_alpha / (e2*params.alpha2_m_beta2),
        c = 1/e - v1_p_alpha*v1_p_alpha / (e*params.alpha2_m_beta2),
        d = -1 + v1_p_alpha*v1_p_alpha*v1_p_alpha / (3*params.alpha2_m_beta2)
    }

    return addCubicFunction(result)
end

local function getLeftCubicFunction(params, e)
    if isLayoutReversed(e) then
        return getPositiveRightCubicFunction(params, e)
    else
        return getPositiveLeftCubicFunction(params, e)
    end
end

local function getRightCubicFunction(params, e)
    if isLayoutReversed(e) then
        return getPositiveLeftCubicFunction(params, e)        
    else
        return getPositiveRightCubicFunction(params, e)
    end
end

local function addLinearFunction(result)
    function result:eval(x)
        return self.a*x + self.b
    end

    function result:evalGrad(x)
        return self.a
    end

    function result:evalGrad2(x)
        return 0
    end

    return result
end

local function getPositiveLeftLinearFunction(params, e)    
    local result = {
        a = -1 / e,
        b = 1
    }

    return addLinearFunction(result)
end

local function getPositiveRightLinearFunction(params, e)    
    local result = {
        a = 1 / e,
        b = -1
    }

    return addLinearFunction(result)
end

local function getLeftLinearFunction(params, e)
    if isLayoutReversed(e) then
        return getPositiveRightLinearFunction(params, e)
    else
        return getPositiveLeftLinearFunction(params, e)
    end
end

local function getRightLinearFunction(params, e)
    if isLayoutReversed(e) then
        return getPositiveLeftLinearFunction(params, e)
    else
        return getPositiveRightLinearFunction(params, e)
    end
end

local function getSquareFuntionBounduaries(params, e)
    if isLayoutReversed(e) then
        return {
            R = e * (1 - params.beta),
            L = e * (1 + params.beta)
        }
    else
        return {
            L = e * (1 - params.beta),
            R = e * (1 + params.beta)
        }
    end
end

local function getSquareFunctionParameters(params, e)
    local result = {
        a = 1 / (e*e*params.alpha_p_beta),
        b = params.q,
        e = e
    }

    function result:eval(x)
        return self.a*(x - self.e)*(x - self.e) + self.b
    end

    function result:evalGrad(x)
        return 2*self.a*(x - self.e)
    end

    function result:evalGrad2(x)
        return 2*self.a
    end

    return result
end

function ErrorCriterionRelative:updateOutput(input, target)
    assert(input:nDimension() == 1 and input:size(1) == 1, "input:nDimension() == 1 and input:size(1) == 1")
    assert(target:nDimension() == 1 and target:size(1) == 1, "target:nDimension() == 1 and target:size(1) == 1")

    local x = input[1]
    local e = target[1]

    assert(e ~= 0, "e != 0")

    local x1 = getLinearFuntionBounduaries(self, e)        
    if x <= x1.L then
        local f = getLeftLinearFunction(self, e)
        return f:eval(x)
    end
    if x >= x1.R then
        local f = getRightLinearFunction(self, e)
        return f:eval(x)
    end

    local x2 = getSquareFuntionBounduaries(self, e)    
    if x >= x2.L and x <= x2.R then
        local f = getSquareFunctionParameters(self, e)        
        return f:eval(x)
    end

    if x < x2.L then        
        local f = getLeftCubicFunction(self, e)        
        return f:eval(x)        
    else
        local f = getRightCubicFunction(self, e)
        return f:eval(x)
    end
end

function ErrorCriterionRelative:updateGradInput(input, target)
    assert(input:nDimension() == 1 and input:size(1) == 1, "input:nDimension() == 1 and input:size(1) == 1")
    assert(target:nDimension() == 1 and target:size(1) == 1, "target:nDimension() == 1 and target:size(1) == 1")

    local x = input[1]
    local e = target[1]

    assert(e ~= 0, "e != 0")

    local x1 = getLinearFuntionBounduaries(self, e)
    if x <= x1.L then
        local f = getLeftLinearFunction(self, e)
        return torch.Tensor{ f:evalGrad(x) }
    end
    if x >= x1.R then
        local f = getRightLinearFunction(self, e)
        return torch.Tensor{ f:evalGrad(x) }
    end    

    local x2 = getSquareFuntionBounduaries(self, e)
    if x >= x2.L and x <= x2.R then
        local f = getSquareFunctionParameters(self, e)
        return torch.Tensor{ f:evalGrad(x) }
    end

    if x < x2.L then
        local f = getLeftCubicFunction(self, e)
        return torch.Tensor{ f:evalGrad(x) }
    else
        local f = getRightCubicFunction(self, e)
        return torch.Tensor{ f:evalGrad(x) }
    end
end

function ErrorCriterionRelative:updateGrad2Input(input, target)
    assert(input:nDimension() == 1 and input:size(1) == 1, "input:nDimension() == 1 and input:size(1) == 1")
    assert(target:nDimension() == 1 and target:size(1) == 1, "target:nDimension() == 1 and target:size(1) == 1")

    local x = input[1]
    local e = target[1]

    assert(e ~= 0, "e != 0")

    local x1 = getLinearFuntionBounduaries(self, e)
    if x <= x1.L then
        local f = getLeftLinearFunction(self, e)
        return torch.Tensor{ f:evalGrad2(x) }
    end
    if x >= x1.R then
        local f = getRightLinearFunction(self, e)
        return torch.Tensor{ f:evalGrad2(x) }
    end    

    local x2 = getSquareFuntionBounduaries(self, e)
    if x >= x2.L and x <= x2.R then
        local f = getSquareFunctionParameters(self, e)
        return torch.Tensor{ f:evalGrad2(x) }
    end

    if x < x2.L then
        local f = getLeftCubicFunction(self, e)
        return torch.Tensor{ f:evalGrad2(x) }
    else
        local f = getRightCubicFunction(self, e)
        return torch.Tensor{ f:evalGrad2(x) }
    end
end

function ErrorCriterionRelative:__tostring__()
  return string.format('%s (alpha:%f, beta:%f)', torch.type(self), self.alpha, self.beta)
end
