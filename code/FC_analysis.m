%% Generate FC matrices

% Load in mb_acc, groupT_ts

% Calculate FC for each mini-block
groupT_rs = reshape(groupT_ts,128,23,[],87);

% For each participant
fc_gen = [];
fc_spec = [];
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
        fc_gen(:,:,jj,ii) = fc_mat(roi_gen==1,:); % 65
        fc_spec(:,:,jj,ii) = fc_mat(roi_spec==1,:); % 54
        %fc_rand(:,:,jj,ii) = fc_mat(roi_gen==0 & roi_spec==0,:); % 363
    end
    fc_all = cat(4, fc_all, fc_mat);
end


%% Functional Connectivity Analysis for Mini-block fingerprinting


% Load in fc_gen.mat, roi_gen.mat, roi_spec.mat, taskset_ord.mat,
% domain_id2.mat

% Get edges between gen and spec
fc_genspec = fc_gen(:,roi_spec==1,:,:);
% Average across 1st and 2nd time and participants
fc_genspec_rs = reshape(fc_genspec,65,54,64,2,87);
fc_genspec_rs = squeeze(mean(fc_genspec_rs,4)); % gen X spec X task id X participants

% Create average FC per rule per domain
% Motor
fc_motor = [];
rule_id = taskset_ord(:,3);
for ii = 1:size(fc_genspec_rs,4)
    sub_fc = fc_genspec_rs(:,:,:,ii);
    for jj = 1:4
        sub_motor(:,:,jj) = mean(sub_fc(:,:,rule_id==jj),3); % gen X spec X domain X rule
    end
    fc_motor = cat(4,fc_motor,sub_motor);
end
% Logic
fc_logic = [];
rule_id = taskset_ord(:,1);
for ii = 1:size(fc_genspec_rs,4)
    sub_fc = fc_genspec_rs(:,:,:,ii);
    for jj = 1:4
        sub_logic(:,:,jj) = mean(sub_fc(:,:,rule_id==jj),3); % gen X spec X domain X rule
    end
    fc_logic = cat(4,fc_logic,sub_logic);
end
% Sensory
fc_sensory = [];
rule_id = taskset_ord(:,2);
for ii = 1:size(fc_genspec_rs,4)
    sub_fc = fc_genspec_rs(:,:,:,ii);
    for jj = 1:4
        sub_sensory(:,:,jj) = mean(sub_fc(:,:,rule_id==jj),3); % gen X spec X domain X rule
    end
    fc_sensory = cat(4,fc_sensory,sub_sensory);
end

% Reorder columns of fc_rule to domain
spec_domain = domain_id2(roi_spec==1);
[spec_domain_ord,idx_spec] = sort(spec_domain,'ascend');
fc_motor_ord = fc_motor(:,idx_spec,:,:);
fc_logic_ord = fc_logic(:,idx_spec,:,:);
fc_sensory_ord = fc_sensory(:,idx_spec,:,:);

% Plot average FC for each pair grouping
mean_fcmotor1 = mean(fc_motor_ord(:,:,1:2,:),[3 4]);
mean_fcmotor2 = mean(fc_motor_ord(:,:,3:4,:),[3 4]);
mean_fclogic1 = mean(fc_logic_ord(:,:,1:3,:),[3 4]);
mean_fclogic2 = mean(fc_logic_ord(:,:,2:4,:),[3 4]);
mean_fcsensory1 = mean(fc_sensory_ord(:,:,1:2,:),[3 4]);
mean_fcsensory2 = mean(fc_sensory_ord(:,:,3:4,:),[3 4]);

g1 = 27.5;
g2 = 27.5+19;
g3 = 27.5+19+8;
data = [mean_fcmotor1 mean_fcmotor2 mean_fclogic1 mean_fclogic2 mean_fcsensory1 mean_fcsensory2];
limits = [min(data,[],'all') max(data,[],'all')];
figure;
subplot(3,2,1);
imagesc(mean_fcmotor1,limits);
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');

subplot(3,2,2);
imagesc(mean_fcmotor2,limits);
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');

subplot(3,2,3);
imagesc(mean_fclogic1,limits);
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');

subplot(3,2,4);
imagesc(mean_fclogic2,limits);
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');

subplot(3,2,5);
imagesc(mean_fcsensory1,limits);
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');

subplot(3,2,6);
imagesc(mean_fcsensory2,limits);
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');

% Plot individually
data_ind = mean_fcsensory2;
figure; imagesc(data_ind,limits);
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');
set(gca,'box','on','FontSize',24,'FontName','Arial','XTickLabel',[], ...
    'XTick',[],'YTickLabel',[],'YTick',[]);
%colorbar;

% Group deltas
fc_mdelta = mean_fcmotor1 - mean_fcmotor2;
fc_ldelta = mean_fclogic1 - mean_fclogic2;
fc_sdelta = mean_fcsensory1 - mean_fcsensory2;

% Plot
g1 = 27.5;
g2 = 27.5+19;
g3 = 27.5+19+8;
data = [fc_mdelta; fc_ldelta; fc_sdelta];
%limits = [min(data,[],'all') max(data,[],'all')];
limits = [-0.1 0.1];
figure;

subplot(1,3,1);
imagesc(fc_mdelta,limits);
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');
colormap(custom());
colorbar;

subplot(1,3,2);
imagesc(fc_ldelta,limits);
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');
colormap(custom());
colorbar;

subplot(1,3,3);
imagesc(fc_sdelta,limits);
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');
colormap(custom());
colorbar;

% Supplementary Figure S3
% Individual plots
figure; imagesc(fc_mdelta,[-0.1 0.1]); colormap(custom());
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');
set(gca,'box','on','FontSize',24,'FontName','Arial','XTickLabel',[], ...
    'XTick',[],'YTickLabel',[],'YTick',[]);

figure; imagesc(fc_ldelta,[-0.1 0.1]); colormap(custom());
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');
set(gca,'box','on','FontSize',24,'FontName','Arial','XTickLabel',[], ...
    'XTick',[],'YTickLabel',[],'YTick',[]);

figure; imagesc(fc_sdelta,[-0.1 0.1]); colormap(custom());
xline([g1 g2 g3],'-','LineWidth',1,'Color','black');
set(gca,'box','on','FontSize',24,'FontName','Arial','XTickLabel',[], ...
    'XTick',[],'YTickLabel',[],'YTick',[]);


% Statistic test between rules
[hm,p] = ttest2(fc_mdelta(:,spec_domain_ord==1),fc_mdelta(:,spec_domain_ord==1.5),'Dim',2); % 64/65
[hl,p] = ttest2(fc_ldelta(:,spec_domain_ord<2),fc_ldelta(:,spec_domain_ord==2.5),'Dim',2); % 56/65
[hs,p] = ttest2(fc_sdelta(:,spec_domain_ord==3),fc_sdelta(:,spec_domain_ord==3.5),'Dim',2); % 60/65

% Conjunction analysis
h_conj = hm.*hl.*hs; % 51/65

% Identify Recombination regions
roi_recomb = zeros(482,1);
roi_recomb(roi_gen==1) = h_conj;


%% Figure 3g

data = roi_recomb;
limits = [min(data) max(data)];
surf_schaef2(data(1:400),limits);
surf_cbm(data(455:482),limits);
subcort_plot(data); colormap(custom());


%% Check ROI stability with bootstrap ratio

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

surf_schaef2(recomb_stab(1:400),[0 1]);
surf_cbm(recomb_stab(455:482),[0 1]);