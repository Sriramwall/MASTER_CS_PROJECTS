import javax.swing.*;

import java.awt.BorderLayout;
import java.util.Date;


public class ReservationFrame extends JFrame{
	
	private ButtonToolbar toolbar;
	private BackButtonToolbar backToolbar;
	private FormPanel formPanel;
	private QueueStatusPanel queueStatusPanel;
	private AppointmentHandler appointmentHandler;
	
	public ReservationFrame() {
		super("Office hours reservation system");
		appointmentHandler = new AppointmentHandler();
		appointmentHandler.GenerateRandomAppointments();
		toolbar = new ButtonToolbar();
		backToolbar = new BackButtonToolbar();
		formPanel = new FormPanel(appointmentHandler);
		queueStatusPanel = new QueueStatusPanel(appointmentHandler);
		
		setLayout(new BorderLayout());
		
		
		backToolbar.setButtonListener(new ButtonListener() {

			@Override
			public void actionEmitted(String action) {
				
				if(action == "next") {
					System.out.println(action);	
					formPanel.DisplayNextAppointment();
					getContentPane().remove(queueStatusPanel);
					getContentPane().remove(backToolbar);
					queueStatusPanel.setVisible(false);
					backToolbar.setVisible(false);
					getContentPane().add(formPanel, BorderLayout.CENTER);	
					getContentPane().add(toolbar, BorderLayout.SOUTH);
					formPanel.setVisible(true);
					toolbar.setVisible(true);
					invalidate();
					validate();
				}
						
			}
			
		});
		
		toolbar.setButtonListener(new ButtonListener() {

			@Override
			public void actionEmitted(String action) {
				
				System.out.println(action);	
				
				if(action == "Present") {					
					appointmentHandler.StartProcessToMarkStudentAsPresent();									
					
				}
				else if(action == "Absent"){
					
					Student currentStudent = appointmentHandler.GetNextAppointment();
					Date currentTime = new Date(System.currentTimeMillis()-10*60*1000);
					if(currentStudent.getAppointmentTime().compareTo(currentTime) < 0) {
						appointmentHandler.StartProcessToMarkStudentAsBan();						
					}
					else {
						appointmentHandler.MoveStudentToTheBack();
					}	
										
				}	
						
				queueStatusPanel.UpdateAllTables();
				getContentPane().remove(formPanel);
				getContentPane().remove(toolbar);
				toolbar.setVisible(false);
				formPanel.setVisible(false);
				getContentPane().add(queueStatusPanel, BorderLayout.CENTER);	
				getContentPane().add(backToolbar, BorderLayout.SOUTH);
				queueStatusPanel.setVisible(true);
				backToolbar.setVisible(true);
				invalidate();
				validate();
						
			}
			
		});
		
		add(formPanel, BorderLayout.CENTER);				
		add(toolbar, BorderLayout.SOUTH);
				
		setSize(600,450);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setVisible(true);
	}
	
	
}
