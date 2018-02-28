library(ggplot2); library(GGally)

iris_data = read.csv("iris.data.txt", header = FALSE)
names(iris_data)<-c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

iris = iris_data
set.seed(100)
samp<-sample(1:150,30)
training<-ir_data[samp,]
testing<-ir_data[-samp,]

# load the package
library(VGAM)
# fit model
fit <- vglm(Species~., family=multinomial, data=training)
# summarize the fit
summary(fit)
# make predictions
probabilities <- predict(fit, testing[,1:4], type="response")
predictions <- apply(probabilities, 1, which.max)
predictions[which(predictions=="1")] <- levels(testing$Species)[1]
predictions[which(predictions=="2")] <- levels(testing$Species)[2]
predictions[which(predictions=="3")] <- levels(testing$Species)[3]
# summarize accuracy
table(predictions, testing$Species)


test <- c('ClaimAccounting_FollowUp','ICD10DiagnosisCodeDictionary','DiagnosisCodeDictionary','PaymentType','Doctor','TaxonomyCode','Encounter','EncounterHistory','TaxonomySpecialty','EncounterDiagnosis','EncounterProcedure','TaxonomyType','EncounterStatus','ICD9ToICD10Crosswalk','InsuranceCompany','InsuranceCompanyPlan','Department','Other','ClaimAccounting_Errors','Patient','PayerTypeCode','Payment','PaymentMethodCode','Practice','PatientCase','ProcedureCodeDictionary','Refund','RefundStatusCode','RefundToPayments','ClaimResponseStatus','PayerScenario','ServiceLocation','InsurancePolicy','Contract','ContractFeeSchedule','InsurancePolicyAuthorization','CapitatedAccount','CapitatedAccountToPayment','ClaimAccounting','ClaimAccounting_Assignments','PatientCaseDate','LastAction','ClaimAccounting_Billings','PayerScenarioType','PaymentClaim','Adjustment','Appointment','AppointmentReason','AppointmentToAppointmentReason','ClaimStateFollowUp','ProcedureCodeRevenueCenterCategory','ProcedureCodeCategory','Claim','ClaimTransaction','ClaimTransactionType')
test2 <- c("Department","DiagnosisCodeDictionary","Doctor","ProcedureCodeCategory","Encounter","EncounterDiagnosis","EncounterProcedure","EncounterStatus","InsuranceCompanyPlan","CapitatedAccount","Other","Patient","CapitatedAccountToPayment","CapitatedAccountToPayment","PayerTypeCode","Payment","ClaimAccounting_Errors","ClaimAccounting_Errors","LastAction","PaymentMethodCode","PayerScenarioType","Practice","ICD10DiagnosisCodeDictionary","ProcedureCodeDictionary","ClaimStateFollowUp","InsuranceCompany","Refund","ProcedureCodeRevenueCenterCategory","RefundStatusCode","RefundToPayments","PatientCaseDate","ServiceLocation","PatientCase","PayerScenario","InsurancePolicy","Adjustment","Appointment","AppointmentReason","TaxonomyCode","ClaimAccounting_FollowUp","ClaimAccounting_FollowUp","AppointmentToAppointmentReason","TaxonomySpecialty","TaxonomySpecialty","TaxonomyType","InsurancePolicyAuthorization","PaymentType","ClaimResponseStatus","PaymentClaim","PaymentClaim","PaymentClaim","ClaimAccounting","Claim","ClaimAccounting_Assignments","ClaimTransaction","ClaimTransactionType","ClaimAccounting_Billings")

test2[!test2 %in% test]
