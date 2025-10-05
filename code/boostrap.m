%% Load in Data

% The following code is used for Supplementary Figure S4 analyses

% Load in mean_Vstab.mat, roi_recomb.mat, roi_spec.mat

% Which ROIs are stable for each latent variable
Vstab_id = abs(mean_Vstab) > 2;

% Which ROIs are stable on top 3 latent variables
Vstab_id2 = Vstab_id(:,1)==1 & Vstab_id(:,2)==1 & Vstab_id(:,3)==1;
sum(Vstab_id2) % 422

% Check whether ROIs of Components and Recombination are stable
sum(roi_spec) % 54
comp_stab = roi_spec.*Vstab_id2;
sum(comp_stab) % 51

sum(roi_recomb) % 51
recomb_stab = roi_recomb.*Vstab_id2;
sum(recomb_stab) % 46

% Check whether ROIs are stable on at least one latent variable
comp_stab_ind = roi_spec.*Vstab_id;
comp_stab_sum = sum(comp_stab_ind(:,1:3),2);
sum(comp_stab_sum>0) % 54

recomb_stab_ind = roi_recomb.*Vstab_id;
recomb_stab_sum = sum(recomb_stab_ind(:,1:3),2);
sum(recomb_stab_sum>0) % 51


%% Plot brain regions

data = comp_stab;
limits = [0 1];
surf_schaef2(data(1:400),limits);
surf_cbm(data(455:482),limits);
subcort_plot(data); colormap(custom());


%% Rerun Time series characteristics using new ROIs

% Load in coef_FIR.mat
ntime = 23;
nroi = 482;
nsub = 87;
coef_FIR2 = reshape(coef_FIR,ntime,[],nroi,nsub);
recomb_FIR = coef_FIR2(:,:,recomb_stab==1,:);
comp_FIR = coef_FIR2(:,:,comp_stab==1,:);


%% Similarity of FIR response across mini-blocks

% Load in mean_corrroi.mat

% Separate out generalised and specialised regions
corr_spec = mean_corrroi(:,comp_stab==1);
corr_recomb = mean_corrroi(:,recomb_stab==1);

mean_corrspec = mean(corr_spec,1)';
mean_corrrecomb = mean(corr_recomb,1)';

% Generalised Linear Mixed Model
data1 = reshape(corr_recomb,[],1);
data2 = reshape(corr_spec,[],1);
data = [data1; data2];
sub_id = [repmat([1:87]',size(corr_recomb,2),1); repmat([1:87]',size(corr_spec,2),1)];
group_id = [2*ones(length(data1),1); 1*ones(length(data2),1)];
% Format variables
tbl = table(data,group_id,sub_id);
tbl.group_id = categorical(tbl.group_id);
tbl.sub_id = categorical(tbl.sub_id);
glme = fitglme(tbl,'data ~ 1 + group_id + (1|sub_id)');
% estimate = 0.12227, SE = 0.0028442, tstat = 42.99, df = 8437, pval = 0,
% CI = [0.1167 0.12785]; p < 1e-16

% Recalculate exact p-values
tval = 42.99;
df = 8437;
pval = 2 * (1 - tcdf(abs(tval), df));


% Boxplot
data1 = mean_corrrecomb;
data2 = mean_corrspec;
g = [ones(length(data1),1); 2*ones(length(data2),1)];
RGB_color = [194, 158, 85;
    110, 143, 169]./255;
figure;
boxplot([data1;data2],g,'Symbol','','OutlierSize',16,'Colors',RGB_color);
hold on
group = [data1; data2];
RGB_color2 = [repmat(RGB_color(1,:),length(data1),1);
    repmat(RGB_color(2,:),length(data2),1)];
scatter(g,group,50,RGB_color2,'filled','jitter','on','jitterAmount',0.1, ...
    'MarkerFaceColor','flat','MarkerFaceAlpha',0.7,'MarkerEdgeColor','flat');
set(gca,'FontSize',24,'FontName','Arial','linew',1.5,'box','off','TickDir','out', ...
    'XTickLabel','');
set(findobj(gca,'type','line'),'linew',3);

% Plot both on same plot
roiR_FIR = mean(recomb_FIR(:,:,2,:),4);
roiC_FIR = mean(comp_FIR(:,:,2,:),4);

figure;
plot(roiR_FIR,'LineWidth',1.5,'Color',[255/255 28/255 99/255 0.75]);
hold on
plot(roiC_FIR,'LineWidth',1.5,'Color',[110/255 143/255 169/255 0.75]);
yline(0,'--','Color',[0 0 0],'LineWidth',1.5);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
xlim([1 23]);
ylim([-2 2]);
hAxes = findobj(gcf,'Type','axes');
hAxes.Position = [0.1300 0.1100 0.5057 0.8028];


%% Dimensionality

% Prep data
nsub = 87;
nroiC = 51;
nroiR = 46;
data1 = reshape(comp_FIR,[],nroiC,nsub); % Time*Task ID X ROI X Participant

% PCA for each participant
dim_exp = [];
for subject = 1:nsub

    data2 = squeeze(data1(:,:,subject)); % Time*TaskID X ROI
    [~,~,~,~,explained,~] = pca(data2);

    dim_exp(subject,1) = (sum(explained)^2)/sum(explained.^2);
end

% Assign to groups
recomb_dim = dim_exp;
comp_dim = dim_exp;

% Boxplot of PC1 comparing between gen and spec
data1 = recomb_dim;
data2 = comp_dim;
g = [ones(length(data1),1); 2*ones(length(data2),1)];
RGB_color = [255, 28, 99;
    110, 143, 169]./255;
figure;
boxplot([data1;data2],g,'Symbol','','OutlierSize',16,'Colors',RGB_color);
hold on
group = [data1; data2];
RGB_color2 = [repmat(RGB_color(1,:),length(data1),1);
    repmat(RGB_color(2,:),length(data2),1)];
scatter(g,group,50,RGB_color2,'filled','jitter','on','jitterAmount',0.1, ...
    'MarkerFaceColor','flat','MarkerFaceAlpha',0.7,'MarkerEdgeColor','flat');
set(gca,'FontSize',24,'FontName','Arial','linew',1.5,'box','off','TickDir','out', ...
    'XTickLabel','');
set(findobj(gca,'type','line'),'linew',3);
%ylim([10 100]);
hAxes = findobj(gcf,'Type','axes');
hAxes.Position = [0.1300 0.1100 0.5057 0.8028];

% Paired t-test comparing dimensionality
data1 = recomb_dim;
data2 = comp_dim;
[h,p,ci,stats] = ttest(data1,data2);
% tstat = -10.6934, df = 86, sd = 1.2455, 95% CI [-1.6933 -1.1624], p <
% 1e-16

% Recalculate exact p-values
tval = -10.6934;
df = 86;
pval = 2 * (1 - tcdf(abs(tval), df));


%% Participation Coefficient

% Load in groupT_ts.mat, recomb_stab.mat, comp_stab.mat

% Calculate FC for each mini-block
groupT_rs = reshape(groupT_ts,128,23,[],87);

% For each participant
fc_all = [];
for ii = 1:size(groupT_rs,4)
    sub_ts = groupT_rs(:,:,:,ii);
    disp(ii);
    fc_mat = [];
    % For each mini-block
    for jj = 1:size(sub_ts,1)
        mb_ts = squeeze(sub_ts(jj,:,:));
        % Calculate functional connectivity
        fc_mat(:,:,jj) = corr(mb_ts);
        % Isolate generalised and specialised regions
        %fc_gen(:,:,jj,ii) = fc_mat(roi_gen==1,:); % 65
        %fc_spec(:,:,jj,ii) = fc_mat(roi_spec==1,:); % 54
        %fc_rand(:,:,jj,ii) = fc_mat(roi_gen==0 & roi_spec==0,:); % 363
    end
    fc_all = cat(4, fc_all, fc_mat);
end

mean_fc = mean(fc_all,[3 4]);

% Finding modules using Louvain algorithm
% Sweep across gamma parameters

% Run using gamma = 1
rng(1234);
data = mean_fc;
ci = [];
q = [];
nodes = size(data,1);
[ci,q] = community_louvain(data,1,1:1:nodes,'negative_asym');

% Calculate participation coefficient
part = participation_coef_sign(data,ci);

% Compare participation coefficient between Recombination and Components
part_recomb = part(recomb_stab==1);
part_comp = part(comp_stab==1);

[h,p,ci,stats] = ttest2(part_recomb,part_comp);
% t = 2.5184, df = 95, sd = 0.0163, p = 0.0135, CI = [0.0018 0.0150]

% Boxplot
data1 = part_recomb;
data2 = part_comp;
g = [ones(length(data1),1); 2*ones(length(data2),1)];
RGB_color = [255, 28, 99;
    110, 143, 169]./255;
figure;
boxplot([data1;data2],g,'Symbol','','OutlierSize',16,'Colors',RGB_color);
hold on
group = [data1; data2];
RGB_color2 = [repmat(RGB_color(1,:),length(data1),1);
    repmat(RGB_color(2,:),length(data2),1)];
scatter(g,group,50,RGB_color2,'filled','jitter','on','jitterAmount',0.1, ...
    'MarkerFaceColor','flat','MarkerFaceAlpha',0.7,'MarkerEdgeColor','flat');
ylim([0.65 0.78]);
set(gca,'FontSize',24,'FontName','Arial','linew',1.5,'box','off','TickDir','out', ...
    'XTickLabel','');
set(findobj(gca,'type','line'),'linew',3);
hAxes = findobj(gcf,'Type','axes');
hAxes.Position = [0.1300 0.1100 0.5057 0.8028];

