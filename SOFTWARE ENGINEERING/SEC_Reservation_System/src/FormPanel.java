import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;

import javax.swing.*;

public class FormPanel extends JPanel{
	
	private AppointmentHandler appointmentHandler;
	private Student currentStudent;
	
	private JLabel studentNameLabel;
	private JLabel studentEmailLabel;
	private JLabel studentQuestionLabel;	
	private JLabel blankLabel1;
	private JLabel blankLabel2;
	
	private JLabel studentNameValueLabel;
	private JLabel studentEmailValueLabel;
	private JLabel studentQuestionValueLabel;
	
	
	public FormPanel(AppointmentHandler handler) {
		Dimension dimension = getPreferredSize();
		dimension.width = 250;
		setPreferredSize(dimension);
		
		appointmentHandler = handler;
		
		studentNameLabel = new JLabel("Name: ");
		studentEmailLabel = new JLabel("Email: ");
		studentQuestionLabel = new JLabel("Question: ");
		blankLabel1 = new JLabel("");
		blankLabel2 = new JLabel("");
		
		studentNameValueLabel = new JLabel("");
		studentEmailValueLabel = new JLabel("");
		studentQuestionValueLabel = new JLabel("");
		
		DisplayNextAppointment();		
		
		setBorder(BorderFactory.createTitledBorder("Student Details"));
		
		setLayout(new GridBagLayout());
		
		GridBagConstraints gc = new GridBagConstraints();	
		
		gc.weightx = 1;
		gc.weighty = 0.5;
		
		gc.gridx = 0;
		gc.gridy = 0;		
		gc.fill = GridBagConstraints.NONE;
		gc.anchor = GridBagConstraints.LINE_START;
		gc.insets = new Insets(0,0,0,0);
		add(blankLabel1, gc);
		
		
		gc.weightx = 1;
		gc.weighty = 0.2;
		
		gc.gridx = 0;
		gc.gridy = 1;
		gc.anchor = GridBagConstraints.LINE_END;
		gc.insets = new Insets(0,0,0,5);
		add(studentNameLabel, gc);
		
		gc.gridx = 1;
		gc.gridy = 1;
		gc.anchor = GridBagConstraints.LINE_START;
		gc.insets = new Insets(0,0,0,0);
		add(studentNameValueLabel, gc);
		
		
		gc.weightx = 1;
		gc.weighty = 0.2;
		
		gc.gridx = 0;
		gc.gridy = 2;
		gc.anchor = GridBagConstraints.LINE_END;
		gc.insets = new Insets(0,0,0,5);
		add(studentEmailLabel, gc);
		
		gc.gridx = 1;
		gc.gridy = 2;
		gc.anchor = GridBagConstraints.LINE_START;
		gc.insets = new Insets(0,0,0,0);
		add(studentEmailValueLabel, gc);
		
		gc.weightx = 1;
		gc.weighty = 0.2;
		
		gc.gridx = 0;
		gc.gridy = 3;
		gc.anchor = GridBagConstraints.LINE_END;
		gc.insets = new Insets(0,0,0,5);
		add(studentQuestionLabel, gc);
		
		gc.gridx = 1;
		gc.gridy = 3;
		gc.anchor = GridBagConstraints.LINE_START;
		gc.insets = new Insets(0,0,0,0);
		add(studentQuestionValueLabel, gc);
		
		gc.weightx = 1;
		gc.weighty = 0.5;
		gc.gridx = 1;
		gc.gridy = 4;
		gc.anchor = GridBagConstraints.LINE_START;
		gc.insets = new Insets(0,0,0,0);
		add(blankLabel2, gc);
	}
	
	public void DisplayNextAppointment() {		
		
		currentStudent = appointmentHandler.GetNextAppointment();
		
		if(currentStudent != null) {			
			studentNameValueLabel.setText(currentStudent.getStudentName());			
			studentEmailValueLabel.setText(currentStudent.getEmail());
			studentQuestionValueLabel.setText(currentStudent.getQuestion());
			
		}
		else {
			JOptionPane.showMessageDialog(this,"There is no student in the queue.");
			studentNameValueLabel.setText("");
			studentEmailValueLabel.setText("");
			studentQuestionValueLabel.setText("");
			System.exit(0);
			
		}		
	}
}
