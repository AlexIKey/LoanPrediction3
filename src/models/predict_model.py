

submission =  pd.DataFrame(data=y_pred,columns=list(le.classes_))
submission.insert(0, 'id', test_data2.id)
submission.reset_index()
submission.to_csv('submission.csv', index=False)