set(0,'DefaultFigureVisible','on');

%% Visualize Data

data = importdata('n90pol.csv');
data = data.data;

amygdala = data(:,1);
acc = data(:,2);
amygdalaation = data(:,3);

pdata = data(:,1:2); 
y = data(:,3); 

figure;
scatter(pdata(y==2,1),pdata(y==2,2), 'r'); hold on;
scatter(pdata(y==3,1),pdata(y==3,2), 'b'); 
scatter(pdata(y==4,1),pdata(y==4,2), 'g'); 
scatter(pdata(y==5,1),pdata(y==5,2), 'm'); 

%% a) 1-D Histograms
figure; 
hist(amygdala); 
title('histogram for amygdala');
figure;
hist(acc);
title('histogram for acc');

%% a) 1-D KDE
figure;
ksdensity(amygdala);
title('KDE for amygdala');

figure;
ksdensity(acc);
title('KDE for acc');

%% b) 2-D Histograms 
figure; 
histogram2(amygdala, acc, 10);
xlabel('amygdala');
ylabel('acc');

%% b 2-D KDE
ksdensity(pdata);
%% c) Independent?
gkde2(pdata); 
kde_xy(amygdala,acc);

%% d) Univariate Conditional Probabilities 
orient_2 = data(data(:,3)==2,:);
orient_3 = data(data(:,3)==3,:);
orient_4 = data(data(:,3)==4,:);
orient_5 = data(data(:,3)==5,:);

figure; 
subplot(2,2,1);
ksdensity(orient_2(1,1:2)); 
title('p(amygdala|orientation=2)');
subplot(2,2,2);
ksdensity(orient_3(1,1:2)); 
title('p(amygdala|orientation=3)');
subplot(2,2,3);
ksdensity(orient_4(1,1:2)); 
title('p(amygdala|orientation=4)');
subplot(2,2,4);
ksdensity(orient_5(1,1:2)); 
title('p(amygdala|orientation=5)');

figure; 
subplot(2,2,1);
ksdensity(orient_2(2,1:2)); 
title('p(acc|orientation=2)');
subplot(2,2,2);
ksdensity(orient_3(2,1:2)); 
title('p(acc|orientation=3)');
subplot(2,2,3);
ksdensity(orient_4(2,1:2)); 
title('p(acc|orientation=4)');
subplot(2,2,4);
ksdensity(orient_5(2,1:2)); 
title('p(acc|orientation=5)');

%% e) Bivariate Conditional Probabilities 

% p(amygdala, acc|orientation = c)
figure; 
ksdensity(orient_2(:,1:2)); 
xlabel('amygdala');
ylabel('acc');
title('p(amygdala,acc|orientation=2)');


%% f) Indepenent? 
figure; 
subplot(2,2,1);
ksdensity(orient_5(:,1:2)); 
xlabel('amygdala');
ylabel('acc');
title('p(amygdala,acc|orientation=5)');
subplot(2,2,2);
kde_xy(orient_5(:,1),orient_5(:,2));
subplot(2,2,3);
gkde2(orient_5(:,1:2));

%% p(x)*p(y) 
function kde_xy(amygdala, acc)
    % Estimate a continuous pdf from the discrete data
    [pdfx xi]= ksdensity(amygdala);
    [pdfy yi]= ksdensity(acc);
    % Create 2-d grid of coordinates and function values, suitable for 3-d plotting
    [xxi,yyi]     = meshgrid(xi,yi);
    [pdfxx,pdfyy] = meshgrid(pdfx,pdfy);
    % Calculate combined pdf, under assumption of independence
    pdfxy = pdfxx.*pdfyy; 
    % Plot the results

    mesh(xxi,yyi,pdfxy)
    xlabel('amygdala');
    ylabel('acc')
    title('p(amygdala|orientation=5)*p(acc|orientation=5)')
    set(gca,'XLim',[min(xi) max(xi)])
    set(gca,'YLim',[min(yi) max(yi)])
end

function contour_xy(amygdala, acc) 
    [pdfx xi]= ksdensity(amygdala);
    [pdfy yi]= ksdensity(acc);
    % Create 2-d grid of coordinates and function values, suitable for 3-d plotting
    [xxi,yyi]     = meshgrid(xi,yi);
    [pdfxx,pdfyy] = meshgrid(pdfx,pdfy);
    % Calculate combined pdf, under assumption of independence
    pdfxy = pdfxx.*pdfyy; 
    % Plot the results

    contour(xxi,yyi,pdfxy)
    xlabel('amygdala');
    ylabel('acc')
    title('contour of p(amygdala)*p(acc)');
    set(gca,'XLim',[min(xi) max(xi)])
    set(gca,'YLim',[min(yi) max(yi)])
end



