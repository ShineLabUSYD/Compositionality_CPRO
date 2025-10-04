%% Get FIR for Components (spec) and Recombination regions

% Load in coef_FIR, roi_spec.mat, roi_recomb.mat
nroi = 482;
nsub = 87;
coef_FIR2 = reshape(coef_FIR,23,[],nroi,nsub);
spec_FIR = coef_FIR2(:,:,roi_spec==1,:); % Time X Task ID X ROI X participant
recomb_FIR = coef_FIR2(:,:,roi_recomb==1,:);


%% Similarity of FIR response across mini-blocks

coef_FIR2 = reshape(coef_FIR,23,64,482,87); % Time X Task ID X ROI X Participant
[ntime,ntasks,nroi,nsub] = size(coef_FIR2);

% For each participant
mean_corrroi = [];
for ii = 1:nsub
    sub_FIR = coef_FIR2(:,:,:,ii);
    % For each region
    for jj = 1:nroi
        disp([ii jj]);
        roi_FIR = sub_FIR(:,:,jj);
        % Compare FIR timeseries across all mini-blocks
        corr_roi = corr(roi_FIR);
        
        % Take lower triangle and average
        template = tril(ones(64)-eye(64));
        % Get the indices of lower triangle
        template_id = find(template);
        % Take lower triangle
        corr_roi_low = corr_roi(template_id);
        % Average correlation across mini-blocks
        mean_corrroi(ii,jj) = mean(corr_roi_low,1);
    end
end

% Separate out generalised and specialised regions
corr_spec = mean_corrroi(:,roi_spec==1);
corr_recomb = mean_corrroi(:,roi_recomb==1);

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
% estimate = 0.1152, SE = 0.0027052, tstat = 42.583, df = 9133, pval = 0,
% CI = [0.10989 0.1205]; p < 1e-16

% Recalculate exact p-values
tval = 42.583;
df = 9133;
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

% Take example generalised/specialised region timeseries across mini-blocks
roiR_FIR = mean(recomb_FIR(:,:,26,:),4);
figure;
plot(roiR_FIR,'LineWidth',1.5,'Color',[194/255 158/255 85/255 0.75]);
yline(0,'--','Color',[0 0 0],'LineWidth',1.5);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
xlim([1 23]);
ylim([-2 2]);

figure;
for ii = 1:size(spec_FIR2,3)
    roiS_FIR = mean(spec_FIR2(:,:,ii,:),4);
    plot(roiS_FIR,'LineWidth',1.5,'Color',[110/255 143/255 169/255 0.75]);
    yline(0,'--','Color',[0 0 0],'LineWidth',1.5);
    set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
    axis('tight');
    xlim([1 23]);
    ylim([-2 2]);
    pause
end

% Figure 4c
% Plot both on same plot
figure;
plot(roiR_FIR,'LineWidth',1.5,'Color',[255/255 28/255 99/255 0.75]);
hold on
roiS_FIR = mean(spec_FIR(:,:,5,:),4);
plot(roiS_FIR,'LineWidth',1.5,'Color',[110/255 143/255 169/255 0.75]);
yline(0,'--','Color',[0 0 0],'LineWidth',1.5);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
xlim([1 23]);
ylim([-2 2]);


%% Dimensionality of regions

% Load in coef_FIR.mat, roi_spec.mat, roi_recomb.mat

% Isolate FIR time series for each group of regions
nroi = 482;
nsub = 87;
spec_FIR = coef_FIR(:,roi_spec==1,:);
recomb_FIR = coef_FIR(:,roi_recomb==1,:);

% Prep data
% Rerun for both Recombination and Spec regions
data1 = recomb_FIR; % Time*Task ID X ROI X Participant
nsub = 87;

% PCA for each participant
exp_var = [];
pc1_var = []; 
dim_exp = [];
for subject = 1:nsub

    data2 = squeeze(data1(:,:,subject)); % Time*TaskID X ROI
    [~,~,~,~,explained,~] = pca(data2);

    exp_var(:,subject) = explained; % PCs X Task ID X Participant
    pc1_var(subject) = explained(1); % Task ID X Participant

    dim_exp(subject,1) = (sum(explained)^2)/sum(explained.^2);
end

% Assign to groups
gen_expvar = exp_var;
gen_pc1 = pc1_var;
gen_dim = dim_exp;

spec_expvar = exp_var;
spec_pc1 = pc1_var;
spec_dim = dim_exp;

% Figure 4a
% Boxplot comparing dimensionality between Spec and Recomb regions
data1 = gen_dim;
data2 = spec_dim;
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
data1 = gen_dim;
data2 = spec_dim;
[h,p,ci,stats] = ttest(data1,data2);
% tstat = -10.6100, df = 86, sd = 1.2818, 95% CI [-1.7312 -1.1849], p <
% 1e-16

% Recalculate exact p-values
tval = -10.6100;
df = 86;
pval = 2 * (1 - tcdf(abs(tval), df));


%% Functional Connectivity Analysis

% Load in mb_acc, groupT_ts

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
    end
    fc_all = cat(4, fc_all, fc_mat);
end

mean_fc = mean(fc_all,[3 4]);

% Finding modules using Louvain algorithm
% Gamma sweep
gamma_sweep = 0.5:0.1:2;
nGamma = 16;
nNodes = 482;
nIter = 100;

ci_iter = zeros(nNodes,nGamma,nIter);
q_iter = zeros(nGamma,nIter);

for nn = 1:nIter
    for gg = 1:nGamma
        [ci_iter(:,gg,nn),q_iter(gg,nn)] = community_louvain(mean_fc,gamma_sweep(gg),[],'negative_asym');
    end
    sprintf('%d',nn)
end

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
part_recomb = part(roi_recomb==1);
part_spec = part(roi_spec==1);

[h,p,ci,stats] = ttest2(part_recomb,part_spec);
% t = 2.9280, df = 103, sd = 0.0166, p = 0.0042, CI = [0.0031 0.0159]

% Figure 4b
% Boxplot comparing participation coefficient between Spec and Recomb
% regions
data1 = part_recomb;
data2 = part_spec;
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


%% Yang Task Variance on coef_FIR

% Load in coef_FIR.mat
ntask = 64;
nroi = 482;
nsub = 87;
ntime = 23;
X = permute(reshape(coef_FIR,ntime,[],nroi,nsub),[2 4 1 3]); % Task ID X nsub X ntime X nroi

mean_X = squeeze(mean(X,2)); % Task ID X ntime X nroi
% For every region
roi_var = [];
for ii = 1:size(mean_X,3)
    roi_FIR = mean_X(:,:,ii);
    % Variance for every time point
    roi_var(ii,:) = var(roi_FIR,[],1); % nroi X ntime
end
mean_var = mean(roi_var,2);

% Visualise on brain
data = mean_var;
limits = [min(mean_var) max(mean_var)];
surf_schaef2(data(1:400),limits);
surf_cbm(data(455:482),limits);
subcort_plot(data); colormap(custom());

% Supplementary Figure S2
% Compare Task variance with PLS and mean FIR measures
% Load in mean_coefFIR.mat total_area.mat
figure;
scatter(mean_var,total_area,30,'MarkerFaceColor','flat',...
    'MarkerFaceAlpha',0.7,'MarkerEdgeColor','flat');
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');
%hAxes = findobj(gcf,'Type','axes');
%hAxes.Position = [0.1300 0.1100 0.5057 0.8028];
xlabel('Mean Task Variance');
ylabel('Rule Bias');