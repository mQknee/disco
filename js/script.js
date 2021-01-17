$(function(){
	$('button').click(function(){
		var user = $('#txtUsername').val();
		var pass = $('#txtPassword').val();
		$.ajax({
			url: '/signUpUser',
			data: $('form').serialize(),
			type: 'POST',
			success: function(response){
				alert(response);
				console.log(response);
			},
			error: function(error){
				alert(error);
				console.log(error);
			}
		});
	});
});
