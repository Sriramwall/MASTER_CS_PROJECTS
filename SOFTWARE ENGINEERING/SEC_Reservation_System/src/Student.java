import java.text.SimpleDateFormat;
import java.util.*;

public class Student {
	
	private int studentID;
	private String studentName;
	private String email;
	private String question;
	private Date appointmentTime = null;
	private boolean isBanned;
	private Date bannedUntilDate;	
    private String status = "Upcoming Appointment";
    
	public String getBannedUntilDate() {
		SimpleDateFormat formatter = new SimpleDateFormat("MM-dd-yyyy");
		String result = "";
		if(bannedUntilDate != null) {
			result = formatter.format(bannedUntilDate);
		}
		else {
			result = "Not banned";
		}
		
		return result;
	}	
	
	public void setBannedUntilDate(Date bannedUntilDate) {
		this.bannedUntilDate = bannedUntilDate;
	}
	public boolean isBanned() {
		return isBanned;
	}
	public void setBanned(boolean isBanned) {
		this.isBanned = isBanned;
	}
	public Date getAppointmentTime() {		
		Random rand = new Random();		

		if(appointmentTime == null) {
			if(rand.nextBoolean()) {					
				setAppointmentTime(new Date(System.currentTimeMillis()-5*60*1000));
			}
			else {
				setAppointmentTime(new Date(System.currentTimeMillis()-11*60*1000));
			}
		}				
		return appointmentTime;
	}
	
	public void setAppointmentTime(Date appointmentTime) {
		this.appointmentTime = appointmentTime;
	}
	public String getEmail() {
		return email;
	}
	public void setEmail(String email) {
		this.email = email;
	}
	public String getStudentName() {
		return studentName;
	}
	public void setStudentName(String studentName) {
		this.studentName = studentName;
	}
	public int getStudentID() {
		return studentID;
	}
	public void setStudentID(int studentID) {
		this.studentID = studentID;
	}
	public String getQuestion() {
		return question;
	}
	public void setQuestion(String question) {
		this.question = question;
	}
	
	public boolean banStudent(boolean ban) {		
		try {
			if(ban) {
				this.isBanned = true;
				this.bannedUntilDate = new Date(System.currentTimeMillis());  
			}
			else {
				this.isBanned = false;
				this.bannedUntilDate = null;
			}
			return true;
		}
		catch(Exception ex) {
			return false;
		}			
	}

	public String getStatus() {
		return status;
	}

	public void setStatus(String status) {
		this.status = status;
	}
	

}
