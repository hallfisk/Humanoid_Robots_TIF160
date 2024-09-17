clf;

constantRegister = [1 -1 -3];
nrOfConstantRegisters = length(constantRegister);
nrOfVariableRegisters = 3;
nrOfOperators = 4; % (+, -, *, /)
nrOfOperands = nrOfVariableRegisters + nrOfConstantRegisters;

% Load data from LoadFunctionData.m
functionData = LoadFunctionData();
x = functionData(:,1);
y = functionData(:,2);
nrOfDataPoints = length(functionData);

globalBestChromosome = matfile('BestChromosome.m').globalBestChromosome;
disp(globalBestChromosome)


fitness = EvaluateIndividual(globalBestChromosome, functionData, constantRegister, nrOfVariableRegisters);
fprintf('The error is %d \n', 1/fitness);

f = CalculateFunction(globalBestChromosome, nrOfVariableRegisters, constantRegister);
fprintf('Estimated function of the data is %s\n', f);

% Plot data and estimation
yEstimate = [];
for k = 1:nrOfDataPoints
    yEstimate(end+1) = CalculateEstimation(globalBestChromosome, x(k), nrOfVariableRegisters, constantRegister);
end

figureHandle = figure(1);
hold on
scatter(x,y);
plot(x, yEstimate,'red','LineWidth',1);
xlabel('x');
ylabel('y');
legend({'Function data points','Obtained function'},'Location','southeast');
