import java.util.*;

public class AppointmentHandler {
	
	private Queue<Student> studentQueue = new LinkedList<>();
	private Queue<Student> banStudentQueue = new LinkedList<>();
	private Queue<Student> attendedStudentQueue = new LinkedList<>();

	public Queue<Student> getStudentQueue() {
		return studentQueue;
	}
	
	public Queue<Student> getBanStudentQueue() {
		return banStudentQueue;
	}
	
	public Queue<Student> getAttendedStudentQueue() {
		return attendedStudentQueue;
	}

	public void setStudentQueue(Queue<Student> studentQueue) {
		this.studentQueue = studentQueue;
	}
	
	public boolean GenerateRandomAppointments() {
		
		try {
			Random rand = new Random();
			boolean addQuestion = true;

			int numberOfReservation = rand.nextInt(5);
			
			for(int i = 0; i < numberOfReservation; i++) {
				Student student = new Student();
				student.setStudentID(i + 1);
				student.setStudentName("Student_" + (i+1));
				if(addQuestion) {
					student.setQuestion("How to solve problem number " + (i+1) + " in the assignment?");
					addQuestion = false;
				}
				else {
					addQuestion = true;
				}
				student.setEmail("Student_" + (i+1) + "@buffalo.edu");		
				studentQueue.add(student);				
			}
			return true;
		}
		catch(Exception ex) {
			return false;
		}
		
	}
	
	public boolean MoveStudentToTheBack() {
		if(studentQueue.size()>0) {
			Student student = studentQueue.poll();
			student.setStatus("Absent");
			studentQueue.add(student);		
			return true;
		}
		else {
			return false;
		}
		
	}
	
	public Student GetNextAppointment() {	
		if(studentQueue.size()>0) {
			return studentQueue.peek();
		}
		else {
			return null;
		}		
	}
	
	public Student GetAndRemoveAppointment() {	
		if(studentQueue.size()>0) {
			return studentQueue.poll();
		}
		else {
			return null;
		}		
	}
	
	public boolean StartProcessToMarkStudentAsPresent() {
		Student student = GetAndRemoveAppointment();
		if(student != null) {
			student.setStatus("Present");
			attendedStudentQueue.add(student);
			return true;
		}
		else {
			return false;
		}
		
	}
	
	public boolean StartProcessToMarkStudentAsBan() {
		Student student = GetAndRemoveAppointment();
		if(student != null) {
			student.banStudent(true);
			student.setStatus("Banned");
			banStudentQueue.add(student);
			return true;
		}
		else {
			return false;
		}
		
	}
	
}
