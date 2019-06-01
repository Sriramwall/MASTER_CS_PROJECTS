import javax.swing.*;
import javax.swing.table.DefaultTableModel;

import java.awt.Dimension;
import java.util.Queue;
import java.util.Vector;

public class QueueStatusPanel extends JPanel{
	
	private AppointmentHandler appointmentHandler;
	
	private JTable upcomingStudentTable;
	private JTable bannedStudentTable;
	private JTable attendedStudentTable;
	private JLabel upcomingStudentLabel = new JLabel("Next in Queue: ");
	private JLabel bannedStudentLabel = new JLabel("Banned: ");
	private JLabel attendedStudentLabel = new JLabel("Appointment completed: ");
	DefaultTableModel model1;
	DefaultTableModel model2;
	DefaultTableModel model3;
	
	public QueueStatusPanel(AppointmentHandler handler) {
				
		appointmentHandler = handler;
		
		InitializeTables1();
		InitializeTables2();
		InitializeTables3();
		
		UpdateAllTables();
		BoxLayout boxlayout = new BoxLayout(this, BoxLayout.Y_AXIS);
		setLayout(boxlayout);
		
		setBorder(BorderFactory.createTitledBorder("Queue Status"));	
		
		add(upcomingStudentLabel);
		add(Box.createRigidArea(new Dimension(0, 5)));
		JScrollPane sp1 = new JScrollPane(upcomingStudentTable); 
        add(sp1);
        add(Box.createRigidArea(new Dimension(0, 15)));
        add(attendedStudentLabel);
		add(Box.createRigidArea(new Dimension(0, 5)));
        JScrollPane sp2 = new JScrollPane(attendedStudentTable); 
        add(sp2);
        add(Box.createRigidArea(new Dimension(0, 15)));
        add(bannedStudentLabel);
		add(Box.createRigidArea(new Dimension(0, 5)));
        JScrollPane sp3 = new JScrollPane(bannedStudentTable); 
        add(sp3);	
		
	}
	
	public void UpdateAllTables() {
		UpdateUpcomingStudentTable();
		UpdateAttendedStudentTable();
		UpdateBannedStudentTable();
	}
	
	public void InitializeTables1() {
		Vector<String> rowOne = new Vector<String>();
	    rowOne.addElement("");
	    rowOne.addElement("");
	    rowOne.addElement("");
	    rowOne.addElement("");
	    rowOne.addElement("");
	    
	    Vector<String> rowTwo = new Vector<String>();
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    
	    Vector<String> rowThree = new Vector<String>();
	    rowThree.addElement("");
	    rowThree.addElement("");
	    rowThree.addElement("");
	    rowThree.addElement("");
	    rowThree.addElement("");
	    
	    Vector<String> rowFour = new Vector<String>();
	    rowFour.addElement("");
	    rowFour.addElement("");
	    rowFour.addElement("");
	    rowFour.addElement("");
	    rowFour.addElement("");
	    
	    Vector<Vector> rowData1 = new Vector<Vector>();
	    rowData1.addElement(rowOne);
	    rowData1.addElement(rowTwo);
	    rowData1.addElement(rowThree);
	    rowData1.addElement(rowFour);
	    
	    Vector<String> columnNames = new Vector<String>();
	    columnNames.addElement("S.No.");
	    columnNames.addElement("Name");
	    columnNames.addElement("Email");
	    columnNames.addElement("Question");
	    columnNames.addElement("Status");
	   
	    model1 = new DefaultTableModel(rowData1, columnNames);
	    
	    upcomingStudentTable = new JTable(model1);
	    
	}
	
	
	public void InitializeTables2() {
		Vector<String> rowOne = new Vector<String>();
	    rowOne.addElement("");
	    rowOne.addElement("");
	    rowOne.addElement("");
	    rowOne.addElement("");
	    rowOne.addElement("");
	    
	    Vector<String> rowTwo = new Vector<String>();
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    
	    Vector<String> rowThree = new Vector<String>();
	    rowThree.addElement("");
	    rowThree.addElement("");
	    rowThree.addElement("");
	    rowThree.addElement("");
	    rowThree.addElement("");
	    
	    Vector<String> rowFour = new Vector<String>();
	    rowFour.addElement("");
	    rowFour.addElement("");
	    rowFour.addElement("");
	    rowFour.addElement("");
	    rowFour.addElement("");
	    
	    Vector<Vector> rowData1 = new Vector<Vector>();
	    rowData1.addElement(rowOne);
	    rowData1.addElement(rowTwo);
	    rowData1.addElement(rowThree);
	    rowData1.addElement(rowFour);
	    
	    Vector<String> columnNames = new Vector<String>();
	    columnNames.addElement("S.No.");
	    columnNames.addElement("Name");
	    columnNames.addElement("Email");
	    columnNames.addElement("Question");
	    columnNames.addElement("Status");
	    	    
	    model2 = new DefaultTableModel(rowData1, columnNames);
	    attendedStudentTable = new JTable(model2);
	    
	}
	
	
	public void InitializeTables3() {
		Vector<String> rowOne = new Vector<String>();
	    rowOne.addElement("");
	    rowOne.addElement("");
	    rowOne.addElement("");
	    rowOne.addElement("");
	    rowOne.addElement("");
	    
	    Vector<String> rowTwo = new Vector<String>();
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    rowTwo.addElement("");
	    
	    Vector<String> rowThree = new Vector<String>();
	    rowThree.addElement("");
	    rowThree.addElement("");
	    rowThree.addElement("");
	    rowThree.addElement("");
	    rowThree.addElement("");
	    
	    Vector<String> rowFour = new Vector<String>();
	    rowFour.addElement("");
	    rowFour.addElement("");
	    rowFour.addElement("");
	    rowFour.addElement("");
	    rowFour.addElement("");
	    
	    Vector<Vector> rowData1 = new Vector<Vector>();
	    rowData1.addElement(rowOne);
	    rowData1.addElement(rowTwo);
	    rowData1.addElement(rowThree);
	    rowData1.addElement(rowFour);
	    
	    Vector<String> bannedColumnNames = new Vector<String>();
	    bannedColumnNames.addElement("S.No.");
	    bannedColumnNames.addElement("Name");
	    bannedColumnNames.addElement("Email");
	    bannedColumnNames.addElement("Stauts");
	    bannedColumnNames.addElement("Banned Until");
	    
	    model3 = new DefaultTableModel(rowData1, bannedColumnNames);
	    
	    bannedStudentTable = new JTable(model3);
	}
	
	
	
	public void UpdateUpcomingStudentTable() {			
		Queue<Student> upcomingStudents = appointmentHandler.getStudentQueue();
		
		int i = 0;
		for (Student student: upcomingStudents) {
			
			model1.setValueAt(Integer.toString(i+1),i,0);
			model1.setValueAt(student.getStudentName(),i,1);
			model1.setValueAt(student.getEmail(),i,2);
			model1.setValueAt(student.getQuestion(),i,3);
			model1.setValueAt(student.getStatus(),i,4);	
			i++;
		}	
		
		while(i<4) {
			model1.setValueAt("",i,0);
			model1.setValueAt("",i,1);
			model1.setValueAt("",i,2);
			model1.setValueAt("",i,3);
			model1.setValueAt("",i,4);	
			i++;			
		}		
	}
	
	public void UpdateAttendedStudentTable() {		
		
		Queue<Student> upcomingStudents = appointmentHandler.getAttendedStudentQueue();
		
		DefaultTableModel model = (DefaultTableModel)attendedStudentTable.getModel();
		
		int i = 0;
		for (Student student: upcomingStudents) {
			
			model2.setValueAt(Integer.toString(i+1),i,0);
			model2.setValueAt(student.getStudentName(),i,1);
			model2.setValueAt(student.getEmail(),i,2);
			model2.setValueAt(student.getQuestion(),i,3);
			model2.setValueAt(student.getStatus(),i,4);	
			i++;
		}	
		
		while(i<4) {
			model2.setValueAt("",i,0);
			model2.setValueAt("",i,1);
			model2.setValueAt("",i,2);
			model2.setValueAt("",i,3);
			model2.setValueAt("",i,4);	
			i++;			
		}	
		
	}
	
	public void UpdateBannedStudentTable() {			
		Queue<Student> upcomingStudents = appointmentHandler.getBanStudentQueue();
				
		int i = 0;
		for (Student student: upcomingStudents) {
			
			model3.setValueAt(Integer.toString(i+1),i,0);
			model3.setValueAt(student.getStudentName(),i,1);
			model3.setValueAt(student.getEmail(),i,2);
			model3.setValueAt(student.getStatus(),i,3);
			model3.setValueAt(student.getBannedUntilDate(),i,4);	
			i++;
		}	
		
		while(i<4) {
			model3.setValueAt("",i,0);
			model3.setValueAt("",i,1);
			model3.setValueAt("",i,2);
			model3.setValueAt("",i,3);
			model3.setValueAt("",i,4);	
			i++;			
		}	
		
	}
	
	
}
