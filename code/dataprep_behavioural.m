%% Load in behavioural data

% Baseline + Rotation
rootdir = 'D:\PhD\compositionality_Cole\behavioural_data'; %set path of root directory
filelist = dir(fullfile(rootdir, '**\*task-cpro*')); %get list of .tsv files for all subfolders

% Create structure to hold the data
storage(length(filelist)) = struct('name',1,'onset',1,'task_event',1, ...
    'rt_in_ms',1,'task_novelty',1,'logic_rule',1, ...
    'sensory_rule',1,'motor_rule',1,'performance',1,'task_id',1,'motor_response',1);

% Load in each subject and store their data
for ii = 1:length(filelist)
    subjectname = extractBefore(filelist(ii).name,'events.tsv'); 
    storage(ii).name = subjectname;
    fullFileName = fullfile(filelist(ii).folder, filelist(ii).name); %create absolute path to tsv file
    tdfread(fullFileName); %read in tsv file
    storage(ii).onset = onset;
    storage(ii).task_event = task_event;
    storage(ii).rt_in_ms = rt_in_ms;
    storage(ii).task_novelty = task_novelty;
    storage(ii).logic_rule = logic_rule;
    storage(ii).sensory_rule = sensory_rule;
    storage(ii).motor_rule = motor_rule;
    storage(ii).performance = performance;
    storage(ii).task_id = task_id;
    storage(ii).motor_response = motor_response;
end

% Change response time and task_id n/a to NaN
for ii = 1:length(storage)
    disp(ii)
    RT = storage(ii).rt_in_ms;
    task_id = storage(ii).task_id;
    double_test = zeros(length(RT),2);
    for jj = 1:length(RT)
        num = zeros(1,2);
        num(1) = str2double(RT(jj,:));
        num(2) = str2double(task_id(jj,:));
        double_test(jj,:) = num;
    end
    storage(ii).rt_in_ms = double_test(:,1);
    storage(ii).task_id = double_test(:,2);
end

% Change Rules to numbers (1-4)
for ii = 1:length(storage)
    temp_rule = zeros(129,4);
    logic = storage(ii).logic_rule;
    sensory = storage(ii).sensory_rule;
    motor = storage(ii).motor_rule;
    task_id = storage(ii).task_id;
    temp_rule(:,4) = task_id;
    for jj = 1:size(temp_rule,1)
        % Logic Rule
        if contains(logic(jj,:),'**BOTH**')
            temp_rule(jj,1) = 1;
        elseif contains(logic(jj,:),'NOT')
            temp_rule(jj,1) = 2;
        elseif contains(logic(jj,:),'*EITHER*')
            temp_rule(jj,1) = 3;
        elseif contains(logic(jj,:),'NEITHER*')
            temp_rule(jj,1) = 4;
        end
        % Sensory Rule
        if contains(sensory(jj,:),'RED')
            temp_rule(jj,2) = 1;
        elseif contains(sensory(jj,:),'VERTICAL')
            temp_rule(jj,2) = 2;
        elseif contains(sensory(jj,:),'PITCH')
            temp_rule(jj,2) = 3;
        elseif contains(sensory(jj,:),'CONSTANT')
            temp_rule(jj,2) = 4;
        end
        % Motor Rule
        if contains(motor(jj,:),'LEFT*INDEX')
            temp_rule(jj,3) = 1;
        elseif contains(motor(jj,:),'LEFT*MIDDLE')
            temp_rule(jj,3) = 2;
        elseif contains(motor(jj,:),'RIGHT*INDEX')
            temp_rule(jj,3) = 3;
        elseif contains(motor(jj,:),'RIGHT*MIDDLE')
            temp_rule(jj,3) = 4;
        end
    end
    storage(ii).rule_set = temp_rule;
end

% Change task events to numbers
% 1 = delay, 2 = encoding, 3 = trial, 4 = ITI
for ii = 1:length(storage)
    temp_event = zeros(129,1);
    task_event = storage(ii).task_event;
    for jj = 1:size(temp_event,1)
        if contains(task_event(jj,:),'delay')
            temp_event(jj) = 1;
        elseif contains(task_event(jj,:),'encoding')
            temp_event(jj) = 2;
        elseif contains(task_event(jj,:),'trial')
            temp_event(jj) = 3;
        elseif contains(task_event(jj,:),'ITI')
            temp_event(jj) = 4;
        end
    end
    storage(ii).task_event2 = temp_event;
end

% DELETE SUBJECT 86 AS PREPROCESSING (FMRIPREP) DIDN'T WORK
storage(393:400) = [];

% Check which mini-blocks were practices for each participant
all_novelty = vertcat(storage.task_novelty);
all_taskid = vertcat(storage.task_id);
prac = [];
a = 1;
for ii = 1:size(all_novelty,1)
    novel = all_novelty(ii,:);
    if strcmp(novel,'Prac ')
        prac(a) = all_taskid(ii);
        a = a + 1;
    end
end
% Check for unique mini-blocks
prac_id = unique(prac,'stable'); % All task ids are practised on


%% Load in timeseries

% Set up folder pathway
myFolder = 'D:\PhD\compositionality_Cole\cpro_timeseries';

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*timeseries_voltron400*.mat');
theFiles = dir(filePattern); % Storing all filenames

%storage(length(theFiles)) = struct('name',1,'ts',1);
% Loop to load in the data, store name and values of zscore
for ii = 1:length(theFiles)
    baseFileName = theFiles(ii).name; 
    fullFileName = fullfile(theFiles(ii).folder, baseFileName); %making absolute path to file
    ts = load(fullFileName); %loads in data
    storage(ii).ts = ts.ts; %stores data under .data
end

% Create ID for correct responses
for ii = 1:length(storage)
    disp(ii)
    performance = storage(ii).performance;
    correct_id = zeros(size(performance,1),1); % number of events per scan
    for jj = 1:size(performance,1)
        if contains(performance(jj,:),'Correct')
            correct_id(jj) = 1;
        end
    end
    storage(ii).correct_id = correct_id;
end


%% Behaviour Analysis

% Load in storage.mat
% Group RT and performance by scan
for ii = 1:length(storage)
    RT = storage(ii).rt_in_ms;
    correct_id = storage(ii).correct_id;
    mb_id = storage(ii).trial_rule(:,4);
    RT2 = RT(RT>=0);
    correct_id2 = correct_id(RT>=0);
    mb_id2 = mb_id(RT>=0);
    behav = [RT2 correct_id2 mb_id2];
    storage(ii).behav = behav;
end

% Check for learning across scans
% Collate all scans from the same subject
% Get unique identifiers for each subject
scan_name = vertcat(storage.name);
% Only keep characters for subject ID
sub_id = {};
for ii = 1:size(scan_name,1)
    sub_id{ii,1} = extractBefore(scan_name(ii,:),'_ses');
end
% Check number of unique IDs
length(unique(sub_id)) % 95
sub_id2 = unique(sub_id);

% Collate across scans
all_RT = [];
all_acc = [];
all_mb = [];
for ii = 1:length(sub_id2)
    RT = [];
    acc = [];
    mb = [];
    for jj = 1:length(storage)
        name = storage(jj).name;
        if contains(name,sub_id2(ii))
            RT = [RT; storage(jj).behav(:,1)];
            acc = [acc; storage(jj).behav(:,2)];
            mb = [mb; storage(jj).behav(:,3)];
        end
    end
    all_RT = [all_RT RT];
    all_acc = [all_acc acc];
    all_mb = [all_mb mb];
end
% Make scan_id for behaviour
scan = 1:8;
all_scan = repelem(scan',48,1);
all_scan = repmat(all_scan,1,87);

% Group-level Behaviour
% Select only mini-blocks with all correct responses
% Load in storage.mat
all_acc_rs = reshape(all_acc,3,[]); % Trial X mini-block*participant
all_mb_rs = reshape(all_mb,3,[]);
all_mb_corr = all_mb_rs;
for ii = 1:size(all_acc_rs,2)
    if ismember(0,all_acc_rs(:,ii))
        all_mb_corr(:,ii) = 0;
    end
end
all_mb_corr = reshape(all_mb_corr,[],95);
all_corr_only = all_mb_corr>0;

% Select response time with correct responses only
all_RT2 = all_RT(all_corr_only==1);
all_scan2 = all_scan(all_corr_only==1);
% Distribution of Response Times
figure; histogram(all_RT2);
% Calculate mean across scans for all participants
mean_RT = zeros(8,1);
for ii = 1:length(unique(all_scan2))
    mean_RT(ii) = mean(all_RT2(all_scan2==ii))/1000; % Convert to seconds
end

% Plot
figure;
plot(mean_RT,'Color',[120 226 213]./255,'LineWidth',1.5);
set(gca,'box','off','FontSize',14,'FontName','Arial','linew',1.5);
axis('tight');
xlabel('Scan');
ylabel('RT (sec)');
ylim([0 2]);

% Calculate Accuracy as a percentage within mini-blocks
all_acc_rs = reshape(all_acc,3,[],95);
all_acc_percent = squeeze(mean(all_acc_rs,1));
scan = 1:8;
all_scan_acc = repelem(scan',16,87);
mean_acc = zeros(8,1);
for ii = 1:8
    mean_acc(ii) = mean(all_acc_percent(all_scan_acc==ii));
end

% Plot
figure;
plot(mean_acc,'Color',[120 226 213]./255,'LineWidth',1.5);
%hold on
%plot(mean_acc(5:8),'Color','red','LineWidth',1.5);
set(gca,'box','off','FontSize',14,'FontName','Arial','linew',1.5);
axis('tight');
xlabel('Scan');
ylabel('Accuracy (%)');
ylim([0 1]);

% Accuracy per mini-block per participant
mb_acc = [];
for ii = 1:size(all_acc,2)
    sub_acc = all_acc(:,ii);
    sub_mb = all_mb(:,ii);
    for jj = 1:length(unique(sub_mb))
        mb_acc(ii,jj) = mean(sub_acc(sub_mb==jj));
    end
end

% Visualise mini-block accuracy
% Mean accuracy per mini-block
mean_mbacc = mean(mb_acc,1)';
figure; bar(mean_mbacc,0.8,'FaceColor','flat','LineWidth',1);
xlabel('Mini-Blocks');
ylabel('Accuracy (%)');
min(mean_mbacc) % 67.37%
max(mean_mbacc) % 97.72%

% Colour bars by rules
% Load in taskset_ord.mat
logic = taskset_ord(:,1);
sensory = taskset_ord(:,2);
motor = taskset_ord(:,3);
colors = [78 185 175; 255 166 0; 249 93 106; 160 81 149]./255;
% Logic
figure;
bL = bar(mean_mbacc,0.8,'FaceColor','flat','LineWidth',1);
bL.CData = colors(logic,:);
title('Logic');
xlabel('Mini-Blocks');
ylabel('Accuracy (%)');
% Sensory
[sensory_ord,idxS] = sort(sensory,'ascend');
mean_mbacc_S = mean_mbacc(idxS);
figure;
bS = bar(mean_mbacc_S,0.8,'FaceColor','flat','LineWidth',1);
bS.CData = colors(sensory_ord,:);
title('Sensory');
xlabel('Mini-Blocks');
ylabel('Accuracy (%)');
% Motor
[motor_ord,idxM] = sort(motor,'ascend');
mean_mbacc_M = mean_mbacc(idxM);
figure;
bM = bar(mean_mbacc_M,0.8,'FaceColor','flat','LineWidth',1);
bM.CData = colors(motor_ord,:);
title('Motor');
xlabel('Mini-Blocks');
ylabel('Accuracy (%)');

% Mean Response Time per mini-block
% Select only mini-blocks with all correct trials
all_mb2 = all_mb(all_corr_only==1);
mean_mbRT = [];
for ii = 1:length(unique(all_mb2))
    mean_mbRT(ii,1) = mean(all_RT2(all_mb2==ii))/1000;
end

% Plot mini-block response time coloured by rules
% Load in taskset_ord.mat
logic = taskset_ord(:,1);
sensory = taskset_ord(:,2);
motor = taskset_ord(:,3);
colors = [78 185 175; 255 166 0; 249 93 106; 160 81 149]./255;
% Logic
figure;
bL2 = bar(mean_mbRT,0.8,'FaceColor','flat','LineWidth',1);
bL2.CData = colors(logic,:);
title('Logic');
xlabel('Mini-Blocks');
ylabel('Response Time (sec)');
% Sensory
[sensory_ord,idxS] = sort(sensory,'ascend');
mean_mbRT_S = mean_mbRT(idxS);
figure;
bS2 = bar(mean_mbRT_S,0.8,'FaceColor','flat','LineWidth',1);
bS2.CData = colors(sensory_ord,:);
title('Sensory');
xlabel('Mini-Blocks');
ylabel('Response Time (sec)');
% Motor
[motor_ord,idxM] = sort(motor,'ascend');
mean_mbRT_M = mean_mbRT(idxM);
figure;
bM2 = bar(mean_mbRT_M,0.8,'FaceColor','flat','LineWidth',1);
bM2.CData = colors(motor_ord,:);
title('Motor');
xlabel('Mini-Blocks');
ylabel('Response Time (sec)');

% Participant-level performance
all_subid = repelem(1:95,384,1);
all_subid2 = all_subid(all_corr_only);
% Mean RT per participant
mean_subRT = [];
for ii = 1:length(unique(all_subid2))
    mean_subRT(ii,1) = mean(all_RT2(all_subid2==ii))/1000;
end
% Barplot of RT per participant
figure; bar(mean_subRT,0.8,'FaceColor','flat','LineWidth',1);
xlabel('Participants');
ylabel('Response Time (sec)');
% Ordered by Response Time
[mean_subRT_ord,idxRT] = sort(mean_subRT,'ascend');
figure; bar(mean_subRT_ord,0.8,'FaceColor','flat','LineWidth',1);
xlabel('Participants');
ylabel('Response Time (sec)');

% Min and Max RT per participant
min_subRT = [];
max_subRT = [];
for ii = 1:length(unique(all_subid2))
    min_subRT(ii,1) = min(all_RT2(all_subid2==ii))/1000;
    max_subRT(ii,1) = max(all_RT2(all_subid2==ii))/1000;
end
% Plot response time in same order as mean response time
%min_subRT_ord = min_subRT(idxRT);
%max_subRT_ord = max_subRT(idxRT);
figure; bar(min_subRT,0.8,'FaceColor','flat','LineWidth',1);
xlabel('Participants');
ylabel('Min Response Time (sec)');
figure; bar(max_subRT,0.8,'FaceColor','flat','LineWidth',1);
xlabel('Participants');
ylabel('Max Response Time (sec)');

% Mean accuracy per participant
mean_subacc = mean(all_acc_percent,1)';
% Order by mean response time
figure; bar(mean_subacc,0.8,'FaceColor','flat','LineWidth',1);
xlabel('Participants');
ylabel('Accuracy (%)');

% Number of incorrect mini-blocks per participant
% Amount of data per participant
% Load in all_mb_corr
all_mb_b = all_mb_corr>0;
sub_nummb = (sum(all_mb_b,1)./3)';

% Plot number of mini-blocks per participant
figure; bar(sub_nummb,0.8,'FaceColor','flat','LineWidth',1);
%xlabel('Participants');
%ylabel('Amount of Mini-blocks');
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');

% Figure
figure; histogram(sub_nummb,15,'FaceColor',[148 157 154]./255,'EdgeColor',[0 0 0]);
xline(65,'--r','LineWidth',3);
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');

% Plot number of mini-blocks for each accuracy
% Load all_acc_percent.mat
figure;
histogram(all_acc_percent,'BinWidth',0.1,'FaceColor',[148 157 154]./255, ...
    'EdgeColor',[0 0 0],'LineWidth',1.5);
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out', ...
    'XTick',[0.05 0.35 0.655 0.955],'XTickLabel','');


% Set threshold for subjects to keep
threshold = 128*0.5; % How much data should each participant have
sub_nummb2 = sub_nummb>threshold; % Threshold data
sum(sub_nummb2) % Check how many participants left: 87

% Plot number of mini-blocks per participant
figure; bar(sub_nummb,0.8,'FaceColor','flat','LineWidth',1);
yline(threshold,'LineWidth',1.5,'Color','red');
xlabel('Participants');
ylabel('Amount of Mini-blocks');

% Find how many participants have at least each mini-block once
num_sub = 95;
unique_mb = zeros(num_sub,1);
for ii = 1:size(all_mb_corr,2)
    sub_mb = all_mb_corr(:,ii);
    unique_mb(ii,1) = sum(unique(sub_mb)>0);
end
% Check how many participants have all mini-blocks
sum(unique_mb==64)
% Visualise number of unique mini-blocks per participant
figure; bar(unique_mb,0.8,'FaceColor','flat','LineWidth',1);
threshold = 64*0.5;
yline(threshold,'LineWidth',1.5,'Color','red');
xlabel('Participants');
ylabel('Amount of Unique Mini-blocks');

% Plot Accuracy and Response Time across scans
% Load in all_acc.mat, all_RT.mat, all_scan2.mat
all_scan_rs = reshape(all_scan2,3,[],87);
all_scan_rs = squeeze(mean(all_scan_rs,1));

mean_acc = [];
std_acc = [];
mean_RT = [];
std_RT = [];
for ii = 1:8
    mean_acc(ii,1) = mean(all_acc_percent(all_scan_rs==ii),1);
    std_acc(ii,1) = std(all_acc_percent(all_scan_rs==ii),[],1);
    mean_RT(ii,1) = mean(all_RT(all_scan2==ii),1);
    std_RT(ii,1) = std(all_RT(all_scan2==ii),[],1);
end

figure; shadedErrorBar(1:8,mean_acc*100,std_acc*100,{'-','color',[222 177 97]./255,'LineWidth',1.5},0);
xlabel(''); ylabel(''); 
ylim([0 100]);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
xlim('tight');

figure; shadedErrorBar(1:8,mean_RT,std_RT);
xlabel(''); ylabel(''); 
ylim([0 2500]);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
xlim('tight');


%% Figure 1d-f

% Mean accuracy across scans per rule per domain
% Load in taskset_ord.mat, all_acc_percent.mat, all_mb_rs.mat, all_scan2.mat,
% all_RT.mat
all_mb_rs = squeeze(mean(reshape(all_mb,3,[],87),1));
% Logic
L1 = taskset_ord(taskset_ord(:,1)==1,4);
L2 = taskset_ord(taskset_ord(:,1)==2,4);
L3 = taskset_ord(taskset_ord(:,1)==3,4);
L4 = taskset_ord(taskset_ord(:,1)==4,4);
L1_acc = [];
L2_acc = [];
L3_acc = [];
L4_acc = [];
for ii = 1:8
    scan_acc = all_acc_percent(all_scan_rs==ii);
    scan_mb = all_mb_rs(all_scan_rs==ii);
    for jj = 1:16
       L1_acc(jj,ii) = mean(scan_acc(scan_mb==L1(jj)),1); 
       L2_acc(jj,ii) = mean(scan_acc(scan_mb==L2(jj)),1); 
       L3_acc(jj,ii) = mean(scan_acc(scan_mb==L3(jj)),1); 
       L4_acc(jj,ii) = mean(scan_acc(scan_mb==L4(jj)),1); 
    end
end
% Plot
figure; 
shadedErrorBar(1:8,mean(L1_acc,1)*100,std(L1_acc,[],1)*100/sqrt(size(L1_acc,1)),{'-','color',[129 75 126]./255,'LineWidth',3},20);
hold on
shadedErrorBar(1:8,mean(L2_acc,1)*100,std(L2_acc,[],1)*100/sqrt(size(L2_acc,1)),{'-','color',[212 151 207]./255,'LineWidth',3},50);
shadedErrorBar(1:8,mean(L3_acc,1)*100,std(L3_acc,[],1)*100/sqrt(size(L3_acc,1)),{'-','color',[159 80 147]./255,'LineWidth',3},50);
shadedErrorBar(1:8,mean(L4_acc,1)*100,std(L4_acc,[],1)*100/sqrt(size(L4_acc,1)),{'-','color',[198 115 192]./255,'LineWidth',3},50);
xlabel(''); ylabel(''); 
ylim([50 100]);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
xlim('tight');

% Sensory
S1 = taskset_ord(taskset_ord(:,2)==1,4);
S2 = taskset_ord(taskset_ord(:,2)==2,4);
S3 = taskset_ord(taskset_ord(:,2)==3,4);
S4 = taskset_ord(taskset_ord(:,2)==4,4);
S1_acc = [];
S2_acc = [];
S3_acc = [];
S4_acc = [];
for ii = 1:8
    scan_acc = all_acc_percent(all_scan_rs==ii);
    scan_mb = all_mb_rs(all_scan_rs==ii);
    for jj = 1:16
       S1_acc(jj,ii) = mean(scan_acc(scan_mb==S1(jj)),1); 
       S2_acc(jj,ii) = mean(scan_acc(scan_mb==S2(jj)),1); 
       S3_acc(jj,ii) = mean(scan_acc(scan_mb==S3(jj)),1); 
       S4_acc(jj,ii) = mean(scan_acc(scan_mb==S4(jj)),1); 
    end
end
% Plot
figure; 
shadedErrorBar(1:8,mean(S1_acc,1)*100,std(S1_acc,[],1)*100/sqrt(size(S1_acc,1)),{'-','color',[214 122 0]./255,'LineWidth',3},20);
hold on
shadedErrorBar(1:8,mean(S2_acc,1)*100,std(S2_acc,[],1)*100/sqrt(size(S2_acc,1)),{'-','color',[255 145 14]./255,'LineWidth',3},50);
shadedErrorBar(1:8,mean(S3_acc,1)*100,std(S3_acc,[],1)*100/sqrt(size(S3_acc,1)),{'-','color',[255 182 138]./255,'LineWidth',3},50);
shadedErrorBar(1:8,mean(S4_acc,1)*100,std(S4_acc,[],1)*100/sqrt(size(S4_acc,1)),{'-','color',[255 219 201]./255,'LineWidth',3},50);
xlabel(''); ylabel(''); 
ylim([50 100]);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
xlim('tight');

% Motor
M1 = taskset_ord(taskset_ord(:,3)==1,4);
M2 = taskset_ord(taskset_ord(:,3)==2,4);
M3 = taskset_ord(taskset_ord(:,3)==3,4);
M4 = taskset_ord(taskset_ord(:,3)==4,4);
M1_acc = [];
M2_acc = [];
M3_acc = [];
M4_acc = [];
for ii = 1:8
    scan_acc = all_acc_percent(all_scan_rs==ii);
    scan_mb = all_mb_rs(all_scan_rs==ii);
    for jj = 1:16
       M1_acc(jj,ii) = mean(scan_acc(scan_mb==M1(jj)),1); 
       M2_acc(jj,ii) = mean(scan_acc(scan_mb==M2(jj)),1); 
       M3_acc(jj,ii) = mean(scan_acc(scan_mb==M3(jj)),1); 
       M4_acc(jj,ii) = mean(scan_acc(scan_mb==M4(jj)),1); 
    end
end
% Plot
figure; 
shadedErrorBar(1:8,mean(M1_acc,1)*100,std(M1_acc,[],1)*100/sqrt(size(M1_acc,1)),{'-','color',[0 136 98]./255,'LineWidth',3},20);
hold on
shadedErrorBar(1:8,mean(M2_acc,1)*100,std(M2_acc,[],1)*100/sqrt(size(M2_acc,1)),{'-','color',[0 167 120]./255,'LineWidth',3},50);
shadedErrorBar(1:8,mean(M3_acc,1)*100,std(M3_acc,[],1)*100/sqrt(size(M3_acc,1)),{'-','color',[0 195 138]./255,'LineWidth',3},50);
shadedErrorBar(1:8,mean(M4_acc,1)*100,std(M4_acc,[],1)*100/sqrt(size(M4_acc,1)),{'-','color',[0 226 159]./255,'LineWidth',3},50);
xlabel(''); ylabel(''); 
ylim([50 100]);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
xlim('tight');


%% Statistical Analyses

% Generalised Linear Mixed Model - Comparing Accuracy across runs comparing per domain
% Load all_acc_percent.mat, all_mb_rs.mat, all_scan_rs.mat, taskset_ord.mat
% Get order of mini-blocks per participant
[mb_ord,idx_mb] = sort(all_mb_rs,'ascend');
% Reorder accuracy and run
all_acc_ord = [];
all_scan_ord = [];
for ii = 1:size(idx_mb,2)
    temp_acc = all_acc_percent(:,ii);
    temp_scan = all_scan_rs(:,ii);
    temp_mb = idx_mb(:,ii);
    all_acc_ord(:,ii) = temp_acc(temp_mb);
    all_scan_ord(:,ii) = temp_scan(temp_mb);
end
all_acc_ord = reshape(all_acc_ord,[],1);
all_scan_ord = reshape(all_scan_ord,[],1);
% Get Rule order
logic_rule = repmat(repelem(taskset_ord(:,1),2,1),87,1);
sensory_rule = repmat(repelem(taskset_ord(:,2),2,1),87,1);
motor_rule = repmat(repelem(taskset_ord(:,3),2,1),87,1);
% Participant ID
sub_id = repelem([1:87]',128,1);
% Instance ID
instance_id = repmat(repmat([1:2]',64,1),87,1);
% Format variables
tbl = table(all_acc_ord,logic_rule,sensory_rule,motor_rule,all_scan_ord,sub_id,instance_id);
tbl.logic_rule = categorical(tbl.logic_rule);
tbl.sensory_rule = categorical(tbl.sensory_rule);
tbl.motor_rule = categorical(tbl.motor_rule);
tbl.sub_id = categorical(tbl.sub_id);
tbl.instance_id = categorical(tbl.instance_id);
% Fit Generalised Linear Mixed Model
glme = fitglme(tbl,['all_acc_ord ~ 1 + logic_rule + sensory_rule + motor_rule + ' ...
    'all_scan_ord + instance_id + (1|sub_id:all_scan_ord)']);

% Recalculate exact p-values
tval = 3.6095;
df = 11124;
pval = 2 * (1 - tcdf(abs(tval), df));


% Generalised Linear Mixed Model - Comparing RT across runs per domain
% Load all_RT.mat, all_mb_rs.mat, all_scan_rs.mat, taskset_ord.mat
% Format RT
all_RT_rs = reshape(all_RT,3,[],87);
RT_mb = squeeze(mean(all_RT_rs,1));
[~,idx_mb] = sort(all_mb_rs,'ascend');
% Reorder accuracy and run
all_RT_ord = [];
all_scan_ord = [];
for ii = 1:size(idx_mb,2)
    temp_RT = RT_mb(:,ii);
    temp_scan = all_scan_rs(:,ii);
    temp_mb = idx_mb(:,ii);
    all_RT_ord(:,ii) = temp_RT(temp_mb);
    all_scan_ord(:,ii) = temp_scan(temp_mb);
end
all_RT_ord = reshape(all_RT_ord,[],1);
all_scan_ord = reshape(all_scan_ord,[],1);
% Get Rule order
logic_rule = repmat(repelem(taskset_ord(:,1),2,1),87,1);
sensory_rule = repmat(repelem(taskset_ord(:,2),2,1),87,1);
motor_rule = repmat(repelem(taskset_ord(:,3),2,1),87,1);
% Participant ID
sub_id = repelem([1:87]',128,1);
% Instance ID
instance_id = repmat(repmat([1:2]',64,1),87,1);
% Format variables
tbl = table(all_RT_ord,logic_rule,sensory_rule,motor_rule,all_scan_ord,sub_id,instance_id);
tbl.logic_rule = categorical(tbl.logic_rule);
tbl.sensory_rule = categorical(tbl.sensory_rule);
tbl.motor_rule = categorical(tbl.motor_rule);
tbl.sub_id = categorical(tbl.sub_id);
tbl.instance_id = categorical(tbl.instance_id);
% Fit Generalised Linear Mixed Model
glme = fitglme(tbl,['all_RT_ord ~ 1 + logic_rule + sensory_rule + motor_rule + ' ...
    'all_scan_ord + instance_id + all_scan_ord*instance_id + (1|sub_id:all_scan_ord)']);

