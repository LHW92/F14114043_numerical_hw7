clc; clear;

% Coefficient matrix A
A = [ 4  -1   0  -1   0   0;
     -1   4  -1   0  -1   0;
      0  -1   4   0   1  -1;
     -1   0   0   4  -1  -1;
      0  -1   0  -1   4  -1;
      0   0  -1   0  -1   4 ];

% Right-hand side vector
b = [0; -1; 9; 4; 8; 6];

% % Coefficient matrix A
% A = [ 4   1   1   0   1;
%      -1  -3   1   1   0;
%       2   1   5  -1  -1;
%      -1  -1  -1   4   0;
%       0   2  -1   1   4];
% 
% % Right-hand side vector
% b = [6; 6; 6; 6; 6];

% Tolerance and max iterations
tol = 1e-6;
maxIter = 1000;

% Initial guess
x0 = zeros(6,1);

%% (a) Jacobi Method
x_jacobi = x0;
D = diag(diag(A));
R = A - D;
for k = 1:maxIter
    x_new = D \ (b - R * x_jacobi);
    if norm(x_new - x_jacobi, inf) < tol
        break;
    end
    x_jacobi = x_new;
end
fprintf("(a) Jacobi Method solution:\n"); disp(x_jacobi);

%% (b) Gauss-Seidel Method
x_gs = x0;
for k = 1:maxIter
    x_old = x_gs;
    for i = 1:length(b)
        x_gs(i) = (b(i) - A(i,1:i-1)*x_gs(1:i-1) - A(i,i+1:end)*x_gs(i+1:end)) / A(i,i);
    end
    if norm(x_gs - x_old, inf) < tol
        break;
    end
end
fprintf("(b) Gauss-Seidel Method solution:\n"); disp(x_gs);

%% (c) SOR Method (ω = 1.25)
omega = 1.25;
x_sor = x0;
for k = 1:maxIter
    x_old = x_sor;
    for i = 1:length(b)
        sigma = A(i,1:i-1)*x_sor(1:i-1) + A(i,i+1:end)*x_sor(i+1:end);
        x_sor(i) = (1 - omega)*x_sor(i) + omega*(b(i) - sigma)/A(i,i);
    end
    if norm(x_sor - x_old, inf) < tol
        break;
    end
end
fprintf("(c) SOR Method solution (ω = %.2f):\n", omega); disp(x_sor);

%% (d) Conjugate Gradient Method
x_cg = x0;
r = b - A*x_cg;
p = r;
for k = 1:maxIter
    Ap = A*p;
    alpha = (r'*r)/(p'*Ap);
    x_new = x_cg + alpha*p;
    r_new = r - alpha*Ap;
    if norm(r_new, inf) < tol
        break;
    end
    beta = (r_new'*r_new)/(r'*r);
    p = r_new + beta*p;
    x_cg = x_new;
    r = r_new;
end
fprintf("(d) Conjugate Gradient Method solution:\n"); disp(x_cg);
