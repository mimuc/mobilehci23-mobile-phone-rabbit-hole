library(dplyr)
library(stringr)
library(ggplot2)
library(ARTool)
library(FSA)
library(corrplot)
library(MBESS)
library(forestplot)
library(likert)

############### This is the file want to push to the repo later (after cleainng up) ###############

# import session with features
all_sessions_with_features <- read.csv("../data/smartphone_sessions_with_features.csv",sep=";")

# TODO does not work at this position:
#all_sessions_with_features <- all_sessions_with_features %>%
#  rowwise() %>% 
#  mutate(n_apps = sum(c_across(starts_with("f_app_category_count_")),na.rm = TRUE)) %>%
#  mutate(n_apps_freq = n_apps/(f_session_length/60)) %>%
#  mutate(f_scrolls = f_scrolls/(f_session_length/60)) %>%
#  mutate(f_clicks = f_clicks/(f_session_length/60))


## ----- Descriptive Statistics: rabbit-hole and usual sessions ------
# RH vs usual sessions
rh_sessions <- all_sessions_with_features %>% filter(target_label=='rabbit_hole')
norh_sessions <- all_sessions_with_features %>% filter(target_label=='no_rabbithole')
unclassifiable_sessions <- all_sessions_with_features %>% filter(!target_label %in% c('rabbit_hole','no_rabbithole'))
print(paste("RH sessions: ",nrow(rh_sessions),", no-RH sessions: ",nrow(norh_sessions),"; unclassifiable sessions: ",nrow(unclassifiable_sessions)))

mean(rh_sessions$count)


# positive vs. negative RH
all_sessions_with_features2 <- all_sessions_with_features %>% 
  mutate(target_label = ifelse(target_label == 'rabbit_hole' & f_esm_regret >= 4,"rabbit_hole_neg",ifelse(target_label == 'rabbit_hole',"rabbit_hole_pos","no_rabbithole")))







## ----- Plots and Graphics -----

#### Figure 5 ####
sub_df <- all_sessions_with_features %>%
  mutate(target_label = ifelse(target_label=="rabbit_hole",1,0)) %>%
  select(where(is.numeric)) %>%
  mutate_if(is.numeric,as.numeric) %>%
  select(-starts_with('f_esm_'))  # remove esm columns, beacuse they contain NAs

sub_df <- sub_df[ , which(apply(sub_df, 2, var) != 0)]

## and histogram:
sub_df2 <- sub_df %>% mutate_at(vars(target_label), as.factor) 
library(plyr)
mu <- ddply(sub_df2, "target_label", summarise, grp.mean=mean(f_session_length, na.rm=TRUE))
sub_df3 <- sub_df2 %>%
  mutate(f_session_length = ifelse(f_session_length < 1,1,f_session_length)) %>%
  mutate(f_session_length = ifelse(f_session_length > 3600,3600,f_session_length)) %>%
  mutate(target_label = ifelse(target_label==1,"rabbit hole","usual session")) %>%
  mutate_at(vars(target_label), as.factor) 
# filter(f_session_length<1800 & f_session_length>1)

ggplot(sub_df3,aes(x=f_session_length,fill=target_label))+
  geom_histogram(aes(y=0.5*..density..),
                 alpha=0.7,position='identity')+
  scale_fill_manual(values=c('#8C0E3F','#00748d'))+
  scale_color_manual(values=c('#8C0E3F','#00748d'))+
  geom_vline(data=mu, aes(xintercept=grp.mean, color=target_label),
             linetype="dashed")+
  scale_x_log10()+
  labs(y = "frequency of sessions", x = "session length in seconds")+
  theme_minimal()+
  theme(
    legend.title = element_blank(),
    legend.position = c(.9, .8)
  )




#### Figure 6 ####
df_sessions_esml <- all_sessions_with_features %>% mutate(across(c("f_esm_track_of_time","f_esm_track_of_space","f_esm_regret","f_esm_agency"), as.factor))
df_sessions_esml <- df_sessions_esml %>% filter(target_label!="")
df_sessions_esml <- df_sessions_esml %>% select(target_label, f_esm_track_of_space, f_esm_track_of_time, f_esm_regret, f_esm_agency)
df_sessions_esml <- df_sessions_esml %>% mutate_at(vars(target_label, f_esm_track_of_space, f_esm_track_of_time, f_esm_regret, f_esm_agency), as.factor) 

fdaff_likert3 <- likert(items=df_sessions_esml[,2:5], grouping=df_sessions_esml[,1])
plot(fdaff_likert3) 




#### Figure 7 #####
## graph apps used absolute:
library(tidyr)
data_apps_long <- all_sessions_with_features %>% 
  filter(!is.na(f_session_length)) %>%
  group_by(studyID,target_label) %>%
  filter(!(abs(f_session_length - median(f_session_length)) > 1.5*sd(f_session_length))) %>%   # fitler outliers with IQR
  ungroup() %>%
  select(session_id,target_label,f_app_category_time_Gaming,f_app_category_time_Health,
         f_app_category_time_System,f_app_category_time_Communication,
         f_app_category_time_News,#f_app_category_time_Food,
         f_app_category_time_Internet,#f_app_category_time_Dating,
         #f_app_category_time_Knowledge,
         f_app_category_time_Visual_Entertainment,
         f_app_category_time_Orientation,f_app_category_time_Social_Media) %>% #select relvant cols
  tidyr::gather(appname,timeinapp, f_app_category_time_Gaming:f_app_category_time_Social_Media, factor_key=TRUE) %>% #wide -> long
  mutate(timeinapp = timeinapp/1000) %>%
  group_by(target_label,appname) %>% dplyr::summarize(mean = mean(timeinapp,na.rm=TRUE),sd=sd(timeinapp,na.rm=TRUE)) %>% #mean und sd berechnen
  mutate(target_label = ifelse(target_label=="rabbit_hole","rabbit hole","usual session")) %>% # make labels more beautiful
  mutate(appname=str_replace(appname,"f_app_category_time_",""))


#tidyr::gather(., key = target_label, value = target_label, c("f_app_category_time_Gaming","f_app_category_time_System"))
### absolute plot:
ggplot(data=data_apps_long, aes(x=appname, y=mean, fill=target_label)) +
  geom_bar(stat="identity", position=position_dodge())+
  #geom_text(aes(label=len), vjust=1.6, color="white",
  #         position = position_dodge(0.9), size=3.5)+
  # geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.2)+
  scale_fill_brewer(palette="Paired")+
  theme_minimal()+
  labs(y = "mean time used per session (s)", x = "app category")+
  theme(
    legend.title = element_blank(),
    #  legend.position = c(.9, .8)
    axis.text.x = element_text(angle = 60, hjust = 1)
  )+
  scale_fill_manual(values=c('#8C0E3F','#00748d'))+
  scale_color_manual(values=c('#8C0E3F','#00748d'))

### relative plot:
data_apps_long_rel <- data_apps_long %>% mutate(mean=ifelse(target_label=="usual session",mean/4.15,mean/16.41))
ggplot(data=data_apps_long_rel, aes(x=appname, y=mean, fill=target_label)) +
  geom_bar(stat="identity", position=position_dodge())+
  #geom_text(aes(label=len), vjust=1.6, color="white",
  #         position = position_dodge(0.9), size=3.5)+
  # geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.2)+
  scale_fill_brewer(palette="Paired")+
  theme_minimal()+
  labs(y = "mean time (s) used per session per minute", x = "app category")+
  theme(
    legend.title = element_blank(),
    #  legend.position = c(.9, .8)
    axis.text.x = element_text(angle = 60, hjust = 1)
  )+
  scale_fill_manual(values=c('#8C0E3F','#00748d'))+
  scale_color_manual(values=c('#8C0E3F','#00748d'))

# data for the description of which apps are used how often
data_apps_long %>% mutate(factor)


#### Figure 9 ####
## time of day
daytimedata <- all_sessions_with_features %>% 
  mutate(target_label = ifelse(target_label=="rabbit_hole","rabbit hole","usual session"))
ggplot(daytimedata,aes(x=f_hour_of_day,fill=target_label))+
  geom_histogram(bins=24,aes(y=0.5*..density..),
                 alpha=0.7,position='identity')+
  scale_fill_manual(values=c('#8C0E3F','#00748d'))+
  scale_color_manual(values=c('#8C0E3F','#00748d'))+
  labs(y = "frequency of sessions", x = "hour of day")+
  theme_minimal()+
  theme(
    legend.title = element_blank(),
    legend.position = c(.3, .7)
  )




